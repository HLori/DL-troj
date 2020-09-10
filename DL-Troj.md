# Environment

- pytorch: 1.5.1  
  - 一定不能是1.6.0 会报错
- 就在我的账户下弄吧 liuhui nsec
- 目录是/DL-troj
- `source activate torchenv` 就是可以用的环境
- 直接运行`main.py` 就好
- 所有数据都在trojai-round0-dataset里 可以copy 一两个模型到/DL-troj/data里面去
- data/example_data 里是id-6, data/data2里是id-1
- 问题：`universalPerturb`里的loss一直不怎么变化 永远是target class保持不变 其余class都不怎么能perturb  我怀疑：
  - 1. 我opt写错了
    2. loss 里面clip的时候 max的值没有设置好（具体看loss1d loss1d2那里）
    3. 我也不知道了



# Struct

- `data\`

  - 可放数据
  - 在 `main.py` 的 `13-14`行被使用到

- `main.py`

  - ```python
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3' # 用到的GPU编号，没人就4块都用
    test_dir = './data/data2/example_data'   # 两个路径
    model_dir = './data/data2/model.pt'
    num_class = 5  # number of labels
    REGU = "l1"  # l1 or l2, regularization of mask loss  不用管
    output_dir = "./results/"  # 好像现在没用 下面都不用管
    # device = ["cuda:1", "cuda:1"]
    # device = [torch.device(d) for d in devices]
    device = "cuda:0"
    device = torch.device(device)
    ```

  - `load_data(root_dir)`: 将文件夹里的img转换成torch，np的格式。 `thres` 是每一类数据的个数，如果太多了CUDA会run out of memory

  - `indices, indices2` 分别是 没有target label的数据索引 和 target label的数据索引， 之后的 `labs, labs2, imgs, imgs2` 分别是label 和 图片数据。 1， 2和上面的意思一样。

  -  ```python
    pert_p, coff_p, ind1, ind2, output_p, log_output, area =\
                UniversalPert(model_dir, batchsize, batchsize2, device).attack(imgs, imgs2, labs, labs2)
    ```

    `pert_p` - mask 大小

     `coff_p` - delta 

     `ind1, ind2`  - 没有被误分类的label（对应imgs和imgs2)

     `output_p` - output of model(imgs+imgs2)

    `log_output` - 和 `[output_p]` 是一样的

    `area` - `[mask 大小]`

  - ```python
    m, det, indic, output = PerImgPert(model_dir, batchsize, device).attack(imgs, labs)
    ```

    `m` - mask

    `det` - delta

    `indic` - 没被攻击成功的indices

    `output` - output of model(imgs)

  - 后面都是计算相似度

- `main_dltnd.py`: 原来 tensorflow 版本的 `main.py`， 如果觉得我写的哪里有错可以参考

- `UniversalPerturb_torch.py`:

  - ```python
    BINARY_SEARCH_STEPS = 4  # number of times to adjust the constant with binary search
    MAX_ITERATIONS = 101  # number of iterations to perform gradient descent
    # ABORT_EARLY = True       # if we stop improving, abort gradient descent early
    LEARNING_RATE = 0.1    # larger values converge faster to less accurate results
    INITIAL_CONST = 0.01     # the initial constant lambda to pick as a first guess
    IMG_SIZE = 224
    CHANNELS = 3
    NUM_LABEL = 5
    ```

    都是可以修改的参数，但是可能修改的主要是 `MAX_ITERATIONS` 和 `LEARNING_RATE`

  - `__init()_`: 初始化 可以不用管

  - `attack(): `  先转换数据格式， 然后通过optim找 untargeted universal perturbation， 对应公式是论文里的 2， 3， 4![image-20200829135534967](C:\Users\40670\AppData\Roaming\Typora\typora-user-images\image-20200829135534967.png)

  - ![image-20200829135605193](C:\Users\40670\AppData\Roaming\Typora\typora-user-images\image-20200829135605193.png)

- `UniversalPerturb.py, PerImagePerturb.py`:  原来TensorFlow版的代码

- `PerImagePerturb_torch.py`:  总体和`UniversalPerturb_torch.py` 相同，但更简单，建议先看这个。

- 其余的都没有用到 可以不用管

