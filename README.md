# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## Task3_5 Results

### Simple

```bash
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05 --PLOT True
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05 --PLOT True
```

/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  9.13240943257366 correct 34 time 5.325505018234253
Epoch  10  loss  4.42007367092547 correct 39 time 1.686577320098877
Epoch  20  loss  2.7118078555320553 correct 44 time 1.5730056762695312
Epoch  30  loss  3.9660211230714673 correct 47 time 1.550971508026123
Epoch  40  loss  2.566437890461556 correct 43 time 2.152249336242676
Epoch  50  loss  1.9837180490397046 correct 49 time 1.5867581367492676
Epoch  60  loss  1.823761455734642 correct 48 time 1.6092689037322998
Epoch  70  loss  1.355840895969357 correct 49 time 2.414971113204956
Epoch  80  loss  1.539825557344594 correct 50 time 1.541485071182251
Epoch  90  loss  0.4812403809122757 correct 49 time 1.5385875701904297
Epoch  100  loss  0.8850697889395405 correct 50 time 2.016721487045288
Epoch  110  loss  0.654566511797433 correct 49 time 1.5333919525146484
Epoch  120  loss  0.547586299895212 correct 49 time 1.5437328815460205
Epoch  130  loss  0.6705636265133067 correct 50 time 1.5968124866485596
Epoch  140  loss  0.34905755645135395 correct 50 time 1.6006312370300293
Epoch  150  loss  0.6495040244588632 correct 50 time 1.647895336151123
Epoch  160  loss  0.39794222155727677 correct 50 time 1.5307319164276123
Epoch  170  loss  0.9990967593345554 correct 50 time 1.7616894245147705
Epoch  180  loss  0.9315423809239955 correct 50 time 1.5390963554382324
Epoch  190  loss  1.0166795215524456 correct 50 time 1.5298607349395752
Epoch  200  loss  0.8249023742945631 correct 50 time 2.176313877105713
Epoch  210  loss  0.8755100220443403 correct 50 time 1.5239672660827637
Epoch  220  loss  0.3381586980675354 correct 50 time 1.546814203262329
Epoch  230  loss  0.6629057759308197 correct 50 time 2.0732061862945557
Epoch  240  loss  0.25079393427380176 correct 50 time 1.6226820945739746
Epoch  250  loss  0.6371741228710984 correct 50 time 1.5250606536865234
Epoch  260  loss  0.6367563245214021 correct 50 time 1.5171253681182861
Epoch  270  loss  0.4462638620308542 correct 50 time 1.5227375030517578
Epoch  280  loss  0.1388406892698936 correct 50 time 1.562476634979248
Epoch  290  loss  0.4209398640212724 correct 50 time 1.5343520641326904
Epoch  300  loss  0.22854362155993346 correct 50 time 1.9637582302093506
Epoch  310  loss  0.48666420809370325 correct 50 time 1.5928986072540283
Epoch  320  loss  0.035661882482528995 correct 50 time 1.622061014175415
Epoch  330  loss  0.7074522704189856 correct 50 time 2.32926607131958
Epoch  340  loss  0.23563159554966656 correct 50 time 1.5318892002105713
Epoch  350  loss  0.30596269005542265 correct 50 time 1.539036512374878
Epoch  360  loss  0.04470034404432113 correct 50 time 1.7973878383636475
Epoch  370  loss  0.11302911853413512 correct 50 time 1.5331618785858154
Epoch  380  loss  0.21064175370020918 correct 50 time 1.523021936416626
Epoch  390  loss  0.2448141304612808 correct 50 time 1.5434160232543945
Epoch  400  loss  0.0718045739467127 correct 50 time 1.761688470840454
Epoch  410  loss  0.1553707599379906 correct 50 time 1.5937199592590332
Epoch  420  loss  0.07029022878108684 correct 50 time 1.5310494899749756
Epoch  430  loss  0.3068125565225177 correct 50 time 2.1475040912628174
Epoch  440  loss  0.19455412099917327 correct 50 time 1.5535264015197754
Epoch  450  loss  0.19733051646216135 correct 49 time 1.5441691875457764
Epoch  460  loss  0.41506243890294137 correct 50 time 1.8780298233032227
Epoch  470  loss  0.10492015771256089 correct 50 time 1.5311031341552734
Epoch  480  loss  0.0699053325914768 correct 50 time 1.5213725566864014
Epoch  490  loss  0.04906769159750647 correct 50 time 1.8574554920196533

real	13m39.775s
user	13m28.720s
sys	0m5.227s
Epoch  0  loss  8.423304610165177 correct 26 time 16.799190521240234
Epoch  10  loss  5.04397830985222 correct 35 time 0.34212541580200195
Epoch  20  loss  5.15239639150631 correct 30 time 0.15265917778015137
Epoch  30  loss  4.113461070834363 correct 44 time 0.15134167671203613
Epoch  40  loss  3.0014650921118955 correct 46 time 0.15349173545837402
Epoch  50  loss  1.8997030727878674 correct 50 time 0.15330934524536133
Epoch  60  loss  2.266947705213484 correct 50 time 0.16245007514953613
Epoch  70  loss  0.8573228554957176 correct 50 time 0.15392684936523438
Epoch  80  loss  1.1204912775390683 correct 50 time 0.1675889492034912
Epoch  90  loss  1.798894421508695 correct 49 time 0.26990699768066406
Epoch  100  loss  1.3096689913807626 correct 47 time 0.16867589950561523
Epoch  110  loss  1.7353673738482243 correct 47 time 0.17962908744812012
Epoch  120  loss  0.7545113293033294 correct 50 time 0.15521907806396484
Epoch  130  loss  1.2887996011809617 correct 50 time 0.15641236305236816
Epoch  140  loss  0.49639064200003785 correct 50 time 0.15237927436828613
Epoch  150  loss  0.5834488057521074 correct 49 time 0.1540682315826416
Epoch  160  loss  0.14266854770422838 correct 50 time 0.15272188186645508
Epoch  170  loss  1.799129349425274 correct 50 time 0.3377809524536133
Epoch  180  loss  0.26106074962937353 correct 50 time 0.15404534339904785
Epoch  190  loss  1.1114429625504623 correct 50 time 0.1557629108428955
Epoch  200  loss  0.15079837393917755 correct 50 time 0.1513981819152832
Epoch  210  loss  0.32480759750824245 correct 50 time 0.16113948822021484
Epoch  220  loss  0.3134297309514967 correct 50 time 0.15211105346679688
Epoch  230  loss  0.17721433390276647 correct 49 time 0.15451550483703613
Epoch  240  loss  0.5938034159935777 correct 50 time 0.29740428924560547
Epoch  250  loss  1.0169597402595405 correct 50 time 0.15481925010681152
Epoch  260  loss  0.3846947524090562 correct 50 time 0.1527547836303711
Epoch  270  loss  0.5988007583237553 correct 49 time 0.15177083015441895
Epoch  280  loss  0.12963656250218056 correct 50 time 0.15388846397399902
Epoch  290  loss  0.5871256605481331 correct 50 time 0.15609240531921387
Epoch  300  loss  0.2410581247914937 correct 50 time 0.16866660118103027
Epoch  310  loss  1.1011316139075271 correct 50 time 0.15470409393310547
Epoch  320  loss  0.2375857333888889 correct 50 time 0.2871367931365967
Epoch  330  loss  0.1314928181320302 correct 50 time 0.15041661262512207
Epoch  340  loss  0.45606802390772505 correct 50 time 0.15247464179992676
Epoch  350  loss  0.3492919767847404 correct 50 time 0.15393686294555664
Epoch  360  loss  0.266920794723668 correct 50 time 0.15334558486938477
Epoch  370  loss  0.6198540236489987 correct 50 time 0.1535332202911377
Epoch  380  loss  0.17934841909644608 correct 50 time 0.15390944480895996
Epoch  390  loss  0.49608499686482266 correct 50 time 0.16588973999023438
Epoch  400  loss  0.19332883770982176 correct 50 time 0.330228328704834
Epoch  410  loss  0.4725824972782552 correct 50 time 0.16768598556518555
Epoch  420  loss  0.37499747801395716 correct 50 time 0.15287232398986816
Epoch  430  loss  0.4856423551321364 correct 50 time 0.15148186683654785
Epoch  440  loss  0.3194645629430578 correct 50 time 0.1541118621826172
Epoch  450  loss  0.16583243355230587 correct 50 time 0.151747465133667
Epoch  460  loss  0.14393748551635263 correct 50 time 0.15389513969421387
Epoch  470  loss  0.22620382767043357 correct 50 time 0.23045611381530762
Epoch  480  loss  0.108450183795745 correct 50 time 0.15334558486938477
Epoch  490  loss  0.09338632098987898 correct 50 time 0.15334582328796387

real	1m42.314s
user	2m8.453s
sys	0m32.581s

### Split

```bash
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05 --PLOT True
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05 --PLOT True
```

split

output

/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  9.13240943257366 correct 34 time 5.325505018234253
Epoch  10  loss  4.42007367092547 correct 39 time 1.686577320098877
Epoch  20  loss  2.7118078555320553 correct 44 time 1.5730056762695312
Epoch  30  loss  3.9660211230714673 correct 47 time 1.550971508026123
Epoch  40  loss  2.566437890461556 correct 43 time 2.152249336242676
Epoch  50  loss  1.9837180490397046 correct 49 time 1.5867581367492676
Epoch  60  loss  1.823761455734642 correct 48 time 1.6092689037322998
Epoch  70  loss  1.355840895969357 correct 49 time 2.414971113204956
Epoch  80  loss  1.539825557344594 correct 50 time 1.541485071182251
Epoch  90  loss  0.4812403809122757 correct 49 time 1.5385875701904297
Epoch  100  loss  0.8850697889395405 correct 50 time 2.016721487045288
Epoch  110  loss  0.654566511797433 correct 49 time 1.5333919525146484
Epoch  120  loss  0.547586299895212 correct 49 time 1.5437328815460205
Epoch  130  loss  0.6705636265133067 correct 50 time 1.5968124866485596
Epoch  140  loss  0.34905755645135395 correct 50 time 1.6006312370300293
Epoch  150  loss  0.6495040244588632 correct 50 time 1.647895336151123
Epoch  160  loss  0.39794222155727677 correct 50 time 1.5307319164276123
Epoch  170  loss  0.9990967593345554 correct 50 time 1.7616894245147705
Epoch  180  loss  0.9315423809239955 correct 50 time 1.5390963554382324
Epoch  190  loss  1.0166795215524456 correct 50 time 1.5298607349395752
Epoch  200  loss  0.8249023742945631 correct 50 time 2.176313877105713
Epoch  210  loss  0.8755100220443403 correct 50 time 1.5239672660827637
Epoch  220  loss  0.3381586980675354 correct 50 time 1.546814203262329
Epoch  230  loss  0.6629057759308197 correct 50 time 2.0732061862945557
Epoch  240  loss  0.25079393427380176 correct 50 time 1.6226820945739746
Epoch  250  loss  0.6371741228710984 correct 50 time 1.5250606536865234
Epoch  260  loss  0.6367563245214021 correct 50 time 1.5171253681182861
Epoch  270  loss  0.4462638620308542 correct 50 time 1.5227375030517578
Epoch  280  loss  0.1388406892698936 correct 50 time 1.562476634979248
Epoch  290  loss  0.4209398640212724 correct 50 time 1.5343520641326904
Epoch  300  loss  0.22854362155993346 correct 50 time 1.9637582302093506
Epoch  310  loss  0.48666420809370325 correct 50 time 1.5928986072540283
Epoch  320  loss  0.035661882482528995 correct 50 time 1.622061014175415
Epoch  330  loss  0.7074522704189856 correct 50 time 2.32926607131958
Epoch  340  loss  0.23563159554966656 correct 50 time 1.5318892002105713
Epoch  350  loss  0.30596269005542265 correct 50 time 1.539036512374878
Epoch  360  loss  0.04470034404432113 correct 50 time 1.7973878383636475
Epoch  370  loss  0.11302911853413512 correct 50 time 1.5331618785858154
Epoch  380  loss  0.21064175370020918 correct 50 time 1.523021936416626
Epoch  390  loss  0.2448141304612808 correct 50 time 1.5434160232543945
Epoch  400  loss  0.0718045739467127 correct 50 time 1.761688470840454
Epoch  410  loss  0.1553707599379906 correct 50 time 1.5937199592590332
Epoch  420  loss  0.07029022878108684 correct 50 time 1.5310494899749756
Epoch  430  loss  0.3068125565225177 correct 50 time 2.1475040912628174
Epoch  440  loss  0.19455412099917327 correct 50 time 1.5535264015197754
Epoch  450  loss  0.19733051646216135 correct 49 time 1.5441691875457764
Epoch  460  loss  0.41506243890294137 correct 50 time 1.8780298233032227
Epoch  470  loss  0.10492015771256089 correct 50 time 1.5311031341552734
Epoch  480  loss  0.0699053325914768 correct 50 time 1.5213725566864014
Epoch  490  loss  0.04906769159750647 correct 50 time 1.8574554920196533

real	13m39.775s
user	13m28.720s
sys	0m5.227s
Epoch  0  loss  8.423304610165177 correct 26 time 16.799190521240234
Epoch  10  loss  5.04397830985222 correct 35 time 0.34212541580200195
Epoch  20  loss  5.15239639150631 correct 30 time 0.15265917778015137
Epoch  30  loss  4.113461070834363 correct 44 time 0.15134167671203613
Epoch  40  loss  3.0014650921118955 correct 46 time 0.15349173545837402
Epoch  50  loss  1.8997030727878674 correct 50 time 0.15330934524536133
Epoch  60  loss  2.266947705213484 correct 50 time 0.16245007514953613
Epoch  70  loss  0.8573228554957176 correct 50 time 0.15392684936523438
Epoch  80  loss  1.1204912775390683 correct 50 time 0.1675889492034912
Epoch  90  loss  1.798894421508695 correct 49 time 0.26990699768066406
Epoch  100  loss  1.3096689913807626 correct 47 time 0.16867589950561523
Epoch  110  loss  1.7353673738482243 correct 47 time 0.17962908744812012
Epoch  120  loss  0.7545113293033294 correct 50 time 0.15521907806396484
Epoch  130  loss  1.2887996011809617 correct 50 time 0.15641236305236816
Epoch  140  loss  0.49639064200003785 correct 50 time 0.15237927436828613
Epoch  150  loss  0.5834488057521074 correct 49 time 0.1540682315826416
Epoch  160  loss  0.14266854770422838 correct 50 time 0.15272188186645508
Epoch  170  loss  1.799129349425274 correct 50 time 0.3377809524536133
Epoch  180  loss  0.26106074962937353 correct 50 time 0.15404534339904785
Epoch  190  loss  1.1114429625504623 correct 50 time 0.1557629108428955
Epoch  200  loss  0.15079837393917755 correct 50 time 0.1513981819152832
Epoch  210  loss  0.32480759750824245 correct 50 time 0.16113948822021484
Epoch  220  loss  0.3134297309514967 correct 50 time 0.15211105346679688
Epoch  230  loss  0.17721433390276647 correct 49 time 0.15451550483703613
Epoch  240  loss  0.5938034159935777 correct 50 time 0.29740428924560547
Epoch  250  loss  1.0169597402595405 correct 50 time 0.15481925010681152
Epoch  260  loss  0.3846947524090562 correct 50 time 0.1527547836303711
Epoch  270  loss  0.5988007583237553 correct 49 time 0.15177083015441895
Epoch  280  loss  0.12963656250218056 correct 50 time 0.15388846397399902
Epoch  290  loss  0.5871256605481331 correct 50 time 0.15609240531921387
Epoch  300  loss  0.2410581247914937 correct 50 time 0.16866660118103027
Epoch  310  loss  1.1011316139075271 correct 50 time 0.15470409393310547
Epoch  320  loss  0.2375857333888889 correct 50 time 0.2871367931365967
Epoch  330  loss  0.1314928181320302 correct 50 time 0.15041661262512207
Epoch  340  loss  0.45606802390772505 correct 50 time 0.15247464179992676
Epoch  350  loss  0.3492919767847404 correct 50 time 0.15393686294555664
Epoch  360  loss  0.266920794723668 correct 50 time 0.15334558486938477
Epoch  370  loss  0.6198540236489987 correct 50 time 0.1535332202911377
Epoch  380  loss  0.17934841909644608 correct 50 time 0.15390944480895996
Epoch  390  loss  0.49608499686482266 correct 50 time 0.16588973999023438
Epoch  400  loss  0.19332883770982176 correct 50 time 0.330228328704834
Epoch  410  loss  0.4725824972782552 correct 50 time 0.16768598556518555
Epoch  420  loss  0.37499747801395716 correct 50 time 0.15287232398986816
Epoch  430  loss  0.4856423551321364 correct 50 time 0.15148186683654785
Epoch  440  loss  0.3194645629430578 correct 50 time 0.1541118621826172
Epoch  450  loss  0.16583243355230587 correct 50 time 0.151747465133667
Epoch  460  loss  0.14393748551635263 correct 50 time 0.15389513969421387
Epoch  470  loss  0.22620382767043357 correct 50 time 0.23045611381530762
Epoch  480  loss  0.108450183795745 correct 50 time 0.15334558486938477
Epoch  490  loss  0.09338632098987898 correct 50 time 0.15334582328796387

real	1m42.314s
user	2m8.453s
sys	0m32.581s


### xor


Script
```bash
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05 --PLOT True
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05 --PLOT True
```

output

/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  6.6730938624007035 correct 40 time 4.868068218231201
Epoch  10  loss  4.781447958335365 correct 43 time 1.6103131771087646
Epoch  20  loss  5.087216780082164 correct 47 time 1.540475606918335
Epoch  30  loss  1.9470386933829014 correct 48 time 1.5528347492218018
Epoch  40  loss  3.235258564936497 correct 47 time 1.5447802543640137
Epoch  50  loss  2.088235202943565 correct 46 time 1.6217048168182373
Epoch  60  loss  2.0911073105819313 correct 49 time 1.6248538494110107
Epoch  70  loss  1.267320775266176 correct 48 time 1.5772182941436768
Epoch  80  loss  1.0386202086399339 correct 48 time 1.5557024478912354
Epoch  90  loss  0.7939638681342109 correct 49 time 1.5739974975585938
Epoch  100  loss  1.7473971772785177 correct 48 time 1.8406431674957275
Epoch  110  loss  1.012479347186361 correct 49 time 1.5466246604919434
Epoch  120  loss  0.9679151125682319 correct 49 time 1.6035339832305908
Epoch  130  loss  1.2434590934147223 correct 49 time 2.619746446609497
Epoch  140  loss  0.8140052101195426 correct 49 time 1.6277832984924316
Epoch  150  loss  1.6763614146162982 correct 49 time 1.6807982921600342
Epoch  160  loss  1.8836081500188797 correct 50 time 2.129786968231201
Epoch  170  loss  0.25703807646678056 correct 50 time 1.5697650909423828
Epoch  180  loss  0.31346044607519696 correct 50 time 1.559337854385376
Epoch  190  loss  1.0956997849934742 correct 50 time 2.281377077102661
Epoch  200  loss  1.1044330494382342 correct 50 time 1.557852029800415
Epoch  210  loss  1.0458642284743396 correct 50 time 1.5393307209014893
Epoch  220  loss  1.4442309051876903 correct 50 time 2.206319808959961
Epoch  230  loss  0.1446966506913943 correct 50 time 1.62687087059021
Epoch  240  loss  0.629023790438118 correct 50 time 1.6149296760559082
Epoch  250  loss  0.21631854369767153 correct 49 time 2.141833543777466
Epoch  260  loss  0.33199522359332023 correct 50 time 1.551999568939209
Epoch  270  loss  0.8754883349825732 correct 50 time 1.5631802082061768
Epoch  280  loss  0.8240066687549792 correct 48 time 1.865671157836914
Epoch  290  loss  0.9246313264482838 correct 50 time 1.5545995235443115
Epoch  300  loss  1.5199368971076965 correct 50 time 1.563666820526123
Epoch  310  loss  0.5846785160374643 correct 50 time 1.808506965637207
Epoch  320  loss  0.23170517456479428 correct 50 time 1.6334469318389893
Epoch  330  loss  0.31183306495369556 correct 50 time 1.5432429313659668
Epoch  340  loss  0.5704288213889264 correct 50 time 2.5094153881073
Epoch  350  loss  0.4701175078804847 correct 50 time 1.5568695068359375
Epoch  360  loss  0.5248332476074474 correct 50 time 1.538431167602539
Epoch  370  loss  0.7966712389356837 correct 50 time 1.6955726146697998
Epoch  380  loss  0.622418268477477 correct 50 time 1.5588710308074951
Epoch  390  loss  0.9059193819514524 correct 50 time 1.5503225326538086
Epoch  400  loss  0.4445998794959335 correct 50 time 1.6184616088867188
Epoch  410  loss  0.14431411051699244 correct 50 time 1.6284847259521484
Epoch  420  loss  0.39283188425318405 correct 50 time 1.5510454177856445
Epoch  430  loss  0.12489831290862108 correct 49 time 1.5441584587097168
Epoch  440  loss  0.21480926873981895 correct 50 time 1.716568946838379
Epoch  450  loss  0.22667528040682108 correct 50 time 1.7235753536224365
Epoch  460  loss  0.3842903664862474 correct 50 time 2.1610171794891357
Epoch  470  loss  0.13780645866512875 correct 50 time 1.5546495914459229
Epoch  480  loss  0.3424174967213922 correct 50 time 1.5525362491607666
Epoch  490  loss  0.28628326762064815 correct 50 time 2.4425997734069824

real	13m52.454s
user	13m42.188s
sys	0m5.297s
Epoch  0  loss  6.709561201033543 correct 26 time 17.118633270263672
Epoch  10  loss  5.36124622887409 correct 42 time 0.15505242347717285
Epoch  20  loss  4.917920917722716 correct 45 time 0.1682271957397461
Epoch  30  loss  5.286333600079285 correct 46 time 0.15489816665649414
Epoch  40  loss  3.7658230558436543 correct 46 time 0.15678095817565918
Epoch  50  loss  3.302714422332472 correct 41 time 0.1555495262145996
Epoch  60  loss  2.6069550204189706 correct 49 time 0.16692495346069336
Epoch  70  loss  1.5188389781857208 correct 42 time 0.15247607231140137
Epoch  80  loss  3.36838818504914 correct 47 time 0.2893052101135254
Epoch  90  loss  3.2160876743818783 correct 46 time 0.15552973747253418
Epoch  100  loss  0.4689450730588733 correct 42 time 0.1667952537536621
Epoch  110  loss  2.538880124182994 correct 49 time 0.15607118606567383
Epoch  120  loss  1.896949192517609 correct 49 time 0.15854549407958984
Epoch  130  loss  2.2208478141032644 correct 49 time 0.15856409072875977
Epoch  140  loss  1.7187051405760558 correct 49 time 0.15448212623596191
Epoch  150  loss  1.954577644052972 correct 50 time 0.21501684188842773
Epoch  160  loss  2.358799751631063 correct 49 time 0.15392112731933594
Epoch  170  loss  1.2068183455451622 correct 49 time 0.15255522727966309
Epoch  180  loss  2.47211177124922 correct 48 time 0.1549379825592041
Epoch  190  loss  0.6426755861399179 correct 49 time 0.17043757438659668
Epoch  200  loss  1.3415310641539446 correct 50 time 0.1538848876953125
Epoch  210  loss  0.5375273271567517 correct 50 time 0.16321539878845215
Epoch  220  loss  1.817504293900146 correct 50 time 0.15512847900390625
Epoch  230  loss  0.8788686864015558 correct 50 time 0.33737635612487793
Epoch  240  loss  1.6122622597448129 correct 49 time 0.1563577651977539
Epoch  250  loss  0.5532310973870066 correct 50 time 0.15279150009155273
Epoch  260  loss  1.3548349442584044 correct 50 time 0.15914249420166016
Epoch  270  loss  1.5408501823907657 correct 50 time 0.15289711952209473
Epoch  280  loss  1.608951301084636 correct 50 time 0.15413951873779297
Epoch  290  loss  0.5734435645937934 correct 49 time 0.1549358367919922
Epoch  300  loss  0.6994186037607915 correct 50 time 0.15519261360168457
Epoch  310  loss  1.0620841335421507 correct 50 time 0.34140753746032715
Epoch  320  loss  0.534522396130441 correct 50 time 0.15898871421813965
Epoch  330  loss  0.14822227059430226 correct 50 time 0.16558170318603516
Epoch  340  loss  1.2128285717640133 correct 50 time 0.16857314109802246
Epoch  350  loss  1.3420132178997874 correct 49 time 0.154127836227417
Epoch  360  loss  0.5466180370682585 correct 50 time 0.15535497665405273
Epoch  370  loss  0.6652102858910854 correct 50 time 0.15595102310180664
Epoch  380  loss  0.4729056916598839 correct 50 time 0.36342954635620117
Epoch  390  loss  0.44456151166518526 correct 50 time 0.1538219451904297
Epoch  400  loss  0.28497932434921824 correct 50 time 0.15312910079956055
Epoch  410  loss  0.246860797460742 correct 50 time 0.1529541015625
Epoch  420  loss  0.42716438486503666 correct 50 time 0.1551671028137207
Epoch  430  loss  0.7855183882218499 correct 50 time 0.15314149856567383
Epoch  440  loss  1.02739277465006 correct 50 time 0.1548004150390625
Epoch  450  loss  0.3681283758432735 correct 50 time 0.16524672508239746
Epoch  460  loss  0.6949056795688212 correct 50 time 0.33641982078552246
Epoch  470  loss  0.28617145549546785 correct 50 time 0.1536424160003662
Epoch  480  loss  0.43323103225183907 correct 50 time 0.1531984806060791
Epoch  490  loss  0.33256515833529116 correct 50 time 0.16982221603393555

real	1m44.230s
user	2m9.407s
sys	0m33.699s


### Bigger Example xor

```bash
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 300 --DATASET xor --RATE 0.01
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 300 --DATASET xor --RATE 0.01
```

/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 19 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 10 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 19 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 10 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 94 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 94 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 10 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 10 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 94 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 19 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 10 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  6.344327273171217 correct 25 time 5.126570701599121
Epoch  10  loss  5.0538160681526145 correct 35 time 1.7708539962768555
Epoch  20  loss  3.6286575601677242 correct 27 time 1.7793333530426025
Epoch  30  loss  3.503455936193459 correct 34 time 1.7795135974884033
Epoch  40  loss  4.6244879835995345 correct 44 time 1.792741298675537
Epoch  50  loss  2.3385863025330855 correct 43 time 2.1169075965881348
Epoch  60  loss  2.46466502636836 correct 44 time 1.911804437637329
Epoch  70  loss  3.7901529062885677 correct 42 time 1.7895402908325195
Epoch  80  loss  4.270144037019738 correct 42 time 1.939164638519287
Epoch  90  loss  3.6297126228126606 correct 42 time 1.791280746459961
Epoch  100  loss  5.1474339143527095 correct 46 time 2.065049886703491
Epoch  110  loss  2.8852551703955616 correct 43 time 1.7621333599090576
Epoch  120  loss  4.043013829999447 correct 47 time 2.176995277404785
Epoch  130  loss  1.8647112084758146 correct 39 time 1.7789702415466309
Epoch  140  loss  1.1901942971224064 correct 43 time 2.311396360397339
Epoch  150  loss  0.9387362737684514 correct 48 time 1.8553597927093506
Epoch  160  loss  2.5006205912585253 correct 46 time 2.426391124725342
Epoch  170  loss  1.0762930005069409 correct 46 time 1.7886157035827637
Epoch  180  loss  2.3016446640123522 correct 44 time 2.539891004562378
Epoch  190  loss  2.665948873693996 correct 47 time 1.7733581066131592
Epoch  200  loss  2.721471394159996 correct 49 time 2.569627046585083
Epoch  210  loss  1.79582938299152 correct 49 time 1.7821636199951172
Epoch  220  loss  1.497112402645081 correct 47 time 2.50431227684021
Epoch  230  loss  2.6889502897067614 correct 45 time 1.8656103610992432
Epoch  240  loss  0.9292296707739738 correct 48 time 3.0526318550109863
Epoch  250  loss  3.6751718542645557 correct 46 time 1.7977755069732666
Epoch  260  loss  2.454086642328835 correct 41 time 2.3194353580474854
Epoch  270  loss  1.7163158362319324 correct 49 time 1.7835478782653809
Epoch  280  loss  0.7492015473377254 correct 45 time 2.1927194595336914
Epoch  290  loss  1.1916739278809245 correct 48 time 1.775536298751831
Epoch  300  loss  4.650561610430547 correct 40 time 2.088425874710083
Epoch  310  loss  3.0434393851935226 correct 47 time 1.8394169807434082
Epoch  320  loss  1.3665022568467045 correct 48 time 2.0934510231018066
Epoch  330  loss  3.100593744147061 correct 45 time 1.7720539569854736
Epoch  340  loss  2.132437054373058 correct 46 time 1.9210071563720703
Epoch  350  loss  1.6077193540941546 correct 46 time 1.825653076171875
Epoch  360  loss  2.12602540411671 correct 42 time 1.8228421211242676
Epoch  370  loss  1.3320408883809827 correct 50 time 1.7761869430541992
Epoch  380  loss  1.560298856376082 correct 49 time 1.830099105834961
Epoch  390  loss  2.3358091132577035 correct 49 time 1.7767012119293213
Epoch  400  loss  1.4337458102313396 correct 48 time 1.8368983268737793
Epoch  410  loss  0.6502873809966513 correct 46 time 1.842139720916748
Epoch  420  loss  1.6121598220675037 correct 50 time 1.7842936515808105
Epoch  430  loss  2.218600826543881 correct 50 time 1.8044898509979248
Epoch  440  loss  0.31090137346001256 correct 50 time 1.774526596069336
Epoch  450  loss  0.3563380596721783 correct 45 time 1.7790846824645996
Epoch  460  loss  2.150187533059133 correct 46 time 1.7812669277191162
Epoch  470  loss  0.9735529810215299 correct 48 time 1.7633745670318604
Epoch  480  loss  0.642057294686772 correct 49 time 1.793694257736206
Epoch  490  loss  1.2395245440037987 correct 50 time 1.8386032581329346

real	15m44.914s
user	15m32.032s
sys	0m6.723s
Epoch  0  loss  9.19600408658375 correct 25 time 17.42084574699402
Epoch  10  loss  2.282184144929384 correct 44 time 0.7679681777954102
Epoch  20  loss  2.410625300604724 correct 47 time 0.762946367263794
Epoch  30  loss  3.221340253468311 correct 43 time 0.7543420791625977
Epoch  40  loss  2.4506451533312608 correct 48 time 0.756948709487915
Epoch  50  loss  1.4448382395633645 correct 48 time 0.9436655044555664
Epoch  60  loss  3.3117529256428138 correct 49 time 0.7587840557098389
Epoch  70  loss  1.3026516867755633 correct 45 time 0.7584290504455566
Epoch  80  loss  2.3653725689390317 correct 49 time 1.4756100177764893
Epoch  90  loss  2.183895830004563 correct 49 time 0.7582333087921143
Epoch  100  loss  2.194358021604166 correct 49 time 0.7624871730804443
Epoch  110  loss  0.8947546284422734 correct 49 time 0.9435479640960693
Epoch  120  loss  1.726241033619279 correct 49 time 0.7456114292144775
Epoch  130  loss  2.015889861300774 correct 49 time 0.7451128959655762
Epoch  140  loss  2.0853853390661 correct 49 time 0.7505745887756348
Epoch  150  loss  1.7203582434012612 correct 49 time 0.7489621639251709
Epoch  160  loss  1.0047492796792137 correct 49 time 0.7573421001434326
Epoch  170  loss  1.6408734267325435 correct 49 time 0.7778470516204834
Epoch  180  loss  0.9989053221236515 correct 49 time 0.7570598125457764
Epoch  190  loss  2.634655031668797 correct 48 time 1.3336575031280518
Epoch  200  loss  1.405028196450336 correct 49 time 0.7788817882537842
Epoch  210  loss  1.9392809546531238 correct 49 time 0.7593958377838135
Epoch  220  loss  1.8397094549337243 correct 49 time 1.0574793815612793
Epoch  230  loss  1.3703777656060057 correct 49 time 0.7707490921020508
Epoch  240  loss  0.9851898596773468 correct 49 time 0.7590103149414062
Epoch  250  loss  1.2087872950720533 correct 50 time 0.758399486541748
Epoch  260  loss  0.8525995495029997 correct 49 time 0.7610328197479248
Epoch  270  loss  0.4538318654130046 correct 49 time 0.7474970817565918
Epoch  280  loss  1.5522388534321157 correct 49 time 0.7477152347564697
Epoch  290  loss  0.37016323110335325 correct 50 time 0.7605838775634766
Epoch  300  loss  1.0194610650007598 correct 49 time 0.8148190975189209
Epoch  310  loss  0.9759845236814392 correct 49 time 0.7558667659759521
Epoch  320  loss  1.1361981047202063 correct 50 time 0.7861547470092773
Epoch  330  loss  1.6421099571340236 correct 49 time 1.3699045181274414
Epoch  340  loss  0.7680122429751826 correct 50 time 0.7554628849029541
Epoch  350  loss  1.4031194847901611 correct 50 time 0.757465124130249
Epoch  360  loss  1.4819509282369738 correct 50 time 0.9049994945526123
Epoch  370  loss  0.9436622913730106 correct 50 time 0.7848954200744629
Epoch  380  loss  0.7017901393459617 correct 50 time 0.7553243637084961
Epoch  390  loss  1.0776264152862252 correct 50 time 0.7660505771636963
Epoch  400  loss  0.5983762786895259 correct 50 time 0.754453182220459
Epoch  410  loss  0.8680406177503555 correct 50 time 0.7430260181427002
Epoch  420  loss  0.8233180857918935 correct 50 time 0.7427637577056885
Epoch  430  loss  1.144548979692052 correct 50 time 0.7471060752868652
Epoch  440  loss  1.0505479047400574 correct 50 time 1.2247951030731201
Epoch  450  loss  1.1948354730058763 correct 50 time 0.7434971332550049
Epoch  460  loss  1.0102271511875631 correct 50 time 0.7606854438781738
Epoch  470  loss  1.1660125814323405 correct 50 time 1.280669927597046
Epoch  480  loss  0.6583830562279109 correct 50 time 0.7613351345062256
Epoch  490  loss  0.7637144449723994 correct 50 time 0.7553634643554688

real	6m59.508s
user	9m52.286s
sys	0m56.346s