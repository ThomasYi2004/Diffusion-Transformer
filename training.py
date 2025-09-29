import jax
from jax import random,numpy as jnp
from flax import linen as nn
import optax
from torch.utils.data import DataLoader
import numpy as np
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

def run_training():
    print("JAX devices:", jax.devices())

    #Define constants:
    BATCH_SIZE = 16
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02

    beta_schedule = jax.numpy.linspace(beta_start, beta_end, T)

    alpha_schedule = 1-beta_schedule
    alpha_prefix_product = np.ones(T+1)
    for i in range(1,1001):
        alpha_prefix_product[i] = alpha_prefix_product[i-1]*alpha_schedule[i]

    alpha_prefix_product = jnp.array(alpha_prefix_product)
    print(alpha_prefix_product)

    #input is CxHxW latent image, transform into patches of size Nx(P^2*C)

    class Patchify(nn.Module):
        p: int

        @nn.compact
        def __call__(self, lat_img):
            B, C, H, W = lat_img.shape
            p = self.p
            assert (H % p == 0 and W % p == 0)
            
            lat_img = lat_img.reshape(B, C, H//p, p, W//p, p)
        
            lat_img = lat_img.transpose(0, 2, 4, 3, 5, 1)
        
            patches = lat_img.reshape(B,-1, p*p*C)
            return patches

    # inp = random.normal(random.PRNGKey(23), (3,4,32,32))

    # pat = jax.vmap(patchify, in_axes=(0,None))(inp, 4)
    # pat = patchify(inp,4)
    # print(pat.shape)

    # def batch_patchify(batch_lat_img, p):
    #     return jax.vmap(patchify, in_axes=(0,None))(batch_lat_img,p)



    #TODO: Positional encoding
    class EmbedPatch(nn.Module):
        # patchdim: int
        embed_dim: int

        def setup(self):
            self.layer = nn.Dense(self.embed_dim)

        # @nn.compact
        def __call__(self, x_t, t):
            return self.layer(x_t)

    #CHATGPT
    class sinusoidal_embedding(nn.Module):
        dim: int

        @nn.compact
        def __call__(self, timesteps):
            dim = self.dim
            half_dim = dim // 2
            freqs = jnp.exp(-jnp.arange(half_dim) * (jnp.log(10000.0) / (half_dim - 1)))
            args = timesteps[:, None] * freqs[None]  # [batch, half_dim]
            embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
            return embedding

    sin = sinusoidal_embedding(dim = 32)
    key = random.PRNGKey(23)
    params = sin.init(key, jnp.array([1,2]))
    output = sin.apply(params, jnp.array([1,2]))
    print(output)
        
    # embed = EmbedPatch(embed_dim=32)
    # key = random.PRNGKey(23)
    # params = embed.init(key, pat)
    # output = embed.apply(params, pat)
    # print(output.shape)

    class MHA(nn.Module):
        num_heads: int
        embed_dim: int

        def setup(self):
            assert self.embed_dim%self.num_heads == 0, "embed_dim not divisible by num_heads"
            self.W = nn.Dense(self.embed_dim)
            self.K = nn.Dense(self.embed_dim)
            self.Q = nn.Dense(self.embed_dim)
            self.W0 = nn.Dense(self.embed_dim)

        def __call__(self, x):
            #Assume x has shape (Batches, Seq_len, embed_dim)
            B,S,_ = x.shape
            q = self.Q(x)
            w = self.W(x)
            k = self.K(x)

            head_dim = self.embed_dim//self.num_heads
            multi_q = q.reshape(B,S,self.num_heads,head_dim).transpose(0,2,1,3)
            multi_w = w.reshape(B,S,self.num_heads,head_dim).transpose(0,2,1,3)
            multi_k = k.reshape(B,S,self.num_heads,head_dim).transpose(0,2,1,3)
            
            attention = jnp.matmul(multi_q, multi_k.transpose(0,1,3,2))/jnp.sqrt(head_dim)
            attention = nn.softmax(attention,-1)
            z = jnp.matmul(attention,multi_w)
            multi_z = self.W0(z.transpose(0,2,1,3).reshape(B,S,self.embed_dim))

            return multi_z

    class DiT_block(nn.Module):
        num_heads: int
        embed_dim: int
        seq_size: int

        def setup(self):
            self.layernorm = nn.LayerNorm()
            self.mha = MHA(num_heads = self.num_heads,embed_dim = self.seq_size)
            self.ffd = nn.Sequential([
                nn.Dense(self.seq_size * 4),
                nn.relu,
                nn.Dense(self.seq_size),
            ])
            self.mlp = nn.Dense(6*self.seq_size)
            

        def __call__(self, x_t, t_emb):
            #x_t.shape: (B,Seq_len, Seq_size)
            activation = x_t
            alpha1,beta1,gamma1,alpha2,beta2,gamma2 = jnp.split(self.mlp(t_emb), 6, axis=1)
            means = jnp.mean(activation, axis=-1)
            variances = jnp.var(activation,axis=-1)
            res = activation
            # activation = self.layernorm(x_t)
            #scale,shift
            activation = (activation-means[:, :, None])/variances[:, :, None]
            print(gamma1.shape)
            activation = (activation*gamma1[:,None,:])+beta1[:,None,:]

            activation = self.mha(activation)
            activation = activation*alpha1[:,None,:]
            activation = activation + res
            res2 = activation
            # activation = self.layernorm(activation)
            means2 = jnp.mean(activation, axis=-1)
            variances2 = jnp.var(activation,axis=-1)
            #scale,shift
            activation = (activation-means2[:, :, None])/variances2[:,:,None]
            activation = (activation*gamma2[:,None,:])+beta2[:,None,:]
            
            activation = self.ffd(activation)
            #scale
            activation = activation*alpha2[:,None,:]
            activation = activation+res2
            return activation
        

    # mha = MHA(4,32)
    # key, key1 = random.split(key)
    # params = mha.init(key1, output) 

    #THIS SECTION IS WRITTEN BY CHATGPT

    # Transforms (standard ImageNet preprocessing)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    data_dir = "/kaggle/input/imagenet100"

    # Collect train shards
    train_folders = [f"train.X{i}" for i in range(1, 2)]

    train_datasets = [
        datasets.ImageFolder(root=folder, transform=transform) 
        for folder in train_folders
    ]

    # Merge into one dataset
    train_dataset = ConcatDataset(train_datasets)

    # Validation dataset
    # val_dataset = datasets.ImageFolder(root=f"{data_dir}/val.X", transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last = True)
    # val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Quick check
    images, labels = next(iter(train_loader))
    print(f"Train batch images: {images.shape}, labels: {labels.shape}")

    # print(images[1][0])
    # patchify = Patchify(p=4)
    # key = random.PRNGKey(23)
    # images = jnp.array(images)
    # params = patchify.init(key, images)
    # output = patchify.apply(params, images)
    # print(images.shape)
    # print(output.shape)

    class Model(nn.Module):
        num_heads: int
        embed_dim: int
        p: int
        n: int

        def setup(self):
            self.patchify = Patchify(self.p)
            self.layernorm = nn.LayerNorm()
            self.dit = DiT_block(num_heads = self.num_heads, embed_dim = self.embed_dim, seq_size=self.p*self.p*3)
            self.sin_embed = sinusoidal_embedding(self.embed_dim)
            
        def __call__(self, x_t, t):
            time_embedding = self.sin_embed(t)
            print(time_embedding.shape)
            activation = self.patchify(x_t)  #shape: BxSeq_lenxSeq_size
            for i in range(self.n):
                activation = self.dit(activation, time_embedding)
            activation = self.layernorm(activation)
            return activation


    # model = Model(12,48,4,1)
    # print(jnp.array(labels)[:,None].shape)
    # key = random.PRNGKey(23)
    # params = model.init(key, jnp.array(images), jnp.array(labels))
    # output = model.apply(params, jnp.array(images), jnp.array(labels))
    # print(output.shape)


    #training loop

    # class MLP(nn.Module):
    #     features: Sequence[int]

    #     @nn.compact
    #     def __call__(self, x):
    #         for feat in self.features[:-1]:
    #             x = nn.relu(nn.Dense(feat)(x))
    #         x = nn.Dense(self.features[-1])(x)
    #         return x    
        
    # model = MLP([16,12,28])
    # params = model.init(key, jnp.ones([1,1,1, 28]))["params"]
    key = random.PRNGKey(23)
    model = Model(12,48,8,1)
    params = model.init(key, jnp.array(images), jnp.array(labels))["params"]

    def loss_fn(params, model, images, labels, eps_true, t):
        # print(eps_true.shape)
        eps_out = model({"params":params},images, t).reshape(eps_true.shape[0],-1)
        eps_true = eps_true.reshape(eps_true.shape[0],-1)
        loss = jnp.linalg.norm(eps_true-eps_out, axis=1)
        # print("LOSS",jnp.mean(loss))
        return jnp.mean(loss)

    @jax.jit
    def train_step(state:TrainState, images, labels, eps_true, t):
        loss_func = lambda params: loss_fn(params, state.apply_fn, images, labels, eps_true, t)
        loss = loss_func(state.params)
        # print(loss)
        loss, grads = jax.value_and_grad(loss_func)(state.params)
        state = state.apply_gradients(grads = grads)
        return loss, state

    NUM_EPOCHS = 3
    lr = 1e-4
    seed = 23
    #

    opt_sgd = optax.adamw(learning_rate=lr, weight_decay=1e-2)
    state = TrainState.create(apply_fn=model.apply, params = params, tx=opt_sgd)
    key = random.PRNGKey(23)
    # X = jax.random.normal(random.PRNGKey(0), (2,3, 28, 28))
    for epoch in range(NUM_EPOCHS):
        # for images, labels in train_loader:
        key, key1 = random.split(key)
        t = jnp.fix(random.uniform(key,shape=BATCH_SIZE)*T+0.99).astype(jnp.int32)
        eps = random.normal(key1, images.shape)
        input_image = jnp.sqrt(alpha_prefix_product[t])[:,None,None,None]*jnp.array(images.numpy())+jnp.sqrt(1-alpha_prefix_product[t])[:,None,None,None]*eps
        loss, state = train_step(state,input_image,0,eps, t)
        print(loss)

if (__name__ == "__main__"):
    run_training()