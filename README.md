## tinytorch


Newest ML framework that you propbaly don't need, <br>
this is really autograd engine backed by numpy<br><br>
#### `tinytorch.py` shall always remain under <b>1000 lines</b>. if not we will <i>revert commit</i>

 
[![Python package](https://github.com/joey00072/tinytorch/actions/workflows/unit_test.yaml/badge.svg)](https://github.com/joey00072/tinytorch/actions/workflows/unit_test.yaml)


$$
f(x) =x^3+x
$$

```python
import tinytorch as tt

def f(x):
  return x**3 + x

x = tt.tensor((tt.arange(700) - 400)/100 , requires_grad=True)
z = f(x)
z.sum().backward()
print(x.grad)

```

<p align="center">
  <img src="images/image-1.png" alt="Alt text" width="70%">
</p>

<p align="center">
  <img src="images/image.png" alt="Alt text">
</p>



### What can you do with it?
#### Automatic diffecrtion, yep
```python 
import tinytorch as tt #👀

def f(x,y):
  return x**2 + x*y + (y**3+y) **0.5

x = tt.rand((5,5), requires_grad=True)
y = tt.rand((5,5), requires_grad=True)
z = f(x,y)
z.sum().backward()
print(x.grad)
print(y.grad)
```

#### Train MNITST, no problemo

```bash
python mnist.py
```

#### GPT?? you bet (yes LLM fr fr)

```bash
GPU=1 python mnist.py
```
note: numpy is too slow to train llm you need to install jax (just using it as faster numpy)
#### Visulization 
If you want to see your computation graph run visulize.py

requirements
```bash
pip install graphviz
sudo apt-get install -y graphviz # IDK what to do for windows I use wsl
```
<p align="center">
  <img src="images/image-2.png"  width="50%" >
</p>


#### why this exists
Bcs I was bored

### DEV BLOG
- Part 1: [pythonstuff/build-tensors](https://www.pythonstuff.com/blog/buinging%20own%20autograd%20engine%20tinytorch%2001) 
- Part 2: [pythonstuff/backward-pass](https://www.pythonstuff.com/blog/buinging%20own%20autograd%20engine%20tinytorch%2001)
- Part 3: [pythonstuff/refactor-&-cleanup](https://www.pythonstuff.com/blog/buinging%20own%20autograd%20engine%20tinytorch%2001)

#### powerlevel
1.0 -  karpathy [micrograd](https://github.com/karpathy/micrograd) (really simple, not much you can do with it) <br>
3.14 -  [tinytorch](https://github.com/joey00072/nanograd) (simpile and you can do lot of things with it) <= ❤️ <br>
69 - [tinygrad](https://github.com/tinygrad/tinygrad) (no longer simple you can do lot more)<br>
∞  -  [pytorch](https://github.com/pytorch/pytorch) (goat library, that makes gpu go burrr)<br>


### contribution guideline
- be nice
- performance optimization / more examples welcome
- doc sources if any
- keep tinytorch.py under 1000 lines

### License
[MIT](./LICENSE)