from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from losses import diversity_loss, consistency_loss


class LMRRS(keras.Model):
    """
    LMR-RS (TensorFlow):
    - user/item ID embeddings -> feature vectors e_u, e_i
    - top-k selection from shared pools
    - scaled dot-product attention for aspect alignment
    """

    def __init__(
            self,
            num_users: int,
            num_items: int,
            dim: int,
            pool_user_size: int,
            pool_item_size: int,
            k: int,
            temperature: float = 1.0,
            lambda_div: float = 1e-4,
            lambda_cons: float = 1e-2,
            div_subsample: int = 512,
            name="LMRRS",
    ):
        super().__init__(name=name)
        self.num_users = num_users
        self.num_items = num_items
        self.d = dim
        self.Nu = pool_user_size
        self.Ni = pool_item_size
        self.k = k
        self.temperature = temperature
        self.lambda_div = lambda_div
        self.lambda_cons = lambda_cons
        self.div_subsample = div_subsample
        self.user_emb = layers.Embedding(num_users, dim, embeddings_initializer="glorot_uniform")
        self.item_emb = layers.Embedding(num_items, dim, embeddings_initializer="glorot_uniform")
        init = tf.keras.initializers.GlorotUniform()
        self.Pu = tf.Variable(init(shape=(self.Nu, self.d)), trainable=True)
        self.Pi = tf.Variable(init(shape=(self.Ni, self.d)), trainable=True)
        self.Wq = layers.Dense(dim, use_bias=False)
        self.Wk = layers.Dense(dim, use_bias=False)
        self.Wv = layers.Dense(dim, use_bias=False)

    @tf.function
    def topk_select(self, feat: tf.Tensor, pool: tf.Tensor):
        temperature = tf.maximum(tf.cast(self.temperature, tf.float32), tf.constant(1e-8, tf.float32))

        feat_exp = tf.expand_dims(feat, axis=1)
        pool_exp = tf.expand_dims(pool, axis=0)
        dist2 = tf.reduce_sum(tf.square(feat_exp - pool_exp), axis=-1)
        logits0 = -dist2 / temperature

        B = tf.shape(feat)[0]
        N = tf.shape(pool)[0]
        neg_inf = tf.constant(-1e9, tf.float32)
        mask = tf.zeros((B, N), dtype=tf.float32)
        selected_list = []
        for _ in tf.range(self.k):
            logits = logits0 + mask * neg_inf
            p_soft = tf.nn.softmax(logits, axis=-1)
            idx = tf.argmax(logits, axis=-1, output_type=tf.int32)
            p_hard = tf.one_hot(idx, depth=N, dtype=tf.float32)
            p_ste = p_hard + tf.stop_gradient(p_soft - p_hard)
            aspect = tf.matmul(p_ste, pool)
            selected_list.append(tf.expand_dims(aspect, axis=1))
            mask += p_hard
        selected = tf.concat(selected_list, axis=1)
        return selected


    @tf.function
    def attention_score(self, user_aspects: tf.Tensor, item_aspects: tf.Tensor):
        Q = self.Wq(item_aspects)
        K = self.Wk(user_aspects)
        V = self.Wv(user_aspects)
        scale = tf.math.rsqrt(tf.cast(self.d, tf.float32))
        logits = tf.matmul(Q, K, transpose_b=True) * scale
        A = tf.nn.softmax(logits, axis=-1)
        U_tilde = tf.matmul(A, V)
        aspect_scores = tf.reduce_sum(U_tilde * item_aspects, axis=-1)
        return tf.reduce_mean(aspect_scores, axis=-1)

    @tf.function
    def score_from_ids(self, u: tf.Tensor, i: tf.Tensor):
        e_u = self.user_emb(u)
        e_i = self.item_emb(i)
        U_sel = self.topk_select(e_u, self.Pu)
        I_sel = self.topk_select(e_i, self.Pi)
        return self.attention_score(U_sel, I_sel)

    def _subsample_pool(self, pool: tf.Tensor):
        N = tf.shape(pool)[0]
        s = tf.minimum(self.div_subsample, N)
        idx = tf.random.shuffle(tf.range(N))[:s]
        return tf.gather(pool, idx)

    @tf.function
    def bpr_loss(self, s_pos: tf.Tensor, s_neg: tf.Tensor):
        return -tf.reduce_mean(tf.math.log_sigmoid(s_pos - s_neg))


    @tf.function
    def pairwise_cosine_matrix(self, x: tf.Tensor):
        x = tf.math.l2_normalize(x, axis=-1)
        return tf.matmul(x, x, transpose_b=True)


    @tf.function
    def diversity_loss(self, pool: tf.Tensor):
        C = self.pairwise_cosine_matrix(pool)
        off = C - tf.linalg.diag(tf.linalg.diag_part(C))
        return tf.reduce_mean(tf.square(off))


    @tf.function
    def consistency_loss(self, feat: tf.Tensor, selected: tf.Tensor):
        sel_mean = tf.reduce_mean(selected, axis=1)
        return tf.reduce_mean(tf.square(feat - sel_mean))


    def train_step(self, data):
        u, pos_i, neg_j = data
        with tf.GradientTape() as tape:
            e_u = self.user_emb(u)
            e_pos = self.item_emb(pos_i)
            e_neg = self.item_emb(neg_j)
            U_sel = self.topk_select(e_u, self.Pu)
            I_sel_pos = self.topk_select(e_pos, self.Pi)
            I_sel_neg = self.topk_select(e_neg, self.Pi)
            s_pos = self.attention_score(U_sel, I_sel_pos)
            s_neg = self.attention_score(U_sel, I_sel_neg)
            bpr = -tf.reduce_mean(tf.math.log_sigmoid(s_pos - s_neg))
            Pu_sub = self._subsample_pool(self.Pu)
            Pi_sub = self._subsample_pool(self.Pi)
            div = diversity_loss(Pu_sub) + diversity_loss(Pi_sub)
            cons = consistency_loss(e_u, U_sel) + consistency_loss(e_pos, I_sel_pos)
            loss = bpr + self.lambda_div * div + self.lambda_cons * cons
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss, "bpr": bpr, "div": div, "cons": cons}
