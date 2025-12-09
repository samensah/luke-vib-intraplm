#!/bin/bash

python src/tsne.py \
    --checkpoints "outputs/lb_refind/-3/checkpoint-1500" \
                  "outputs/lb_refind/-2/checkpoint-1500" \
                  "outputs/lb_refind/0/checkpoint-1500" \
                  "outputs/lb_refind/6/checkpoint-1500" \
                  "outputs/lb_refind/-1/checkpoint-1500" \
    --data_file data/refind/test_cf.json \
    --device cuda


# python src/tsne.py \
#     --checkpoints "outputs/lb_refind/-3/checkpoint-1500" \
#                  "outputs/lb_refind/-2/checkpoint-1500" \
#                  "outputs/lb_refind/0/checkpoint-1500" \
#                  "outputs/lb_refind/1/checkpoint-1500" \
#                  "outputs/lb_refind/2/checkpoint-1500" \
#                  "outputs/lb_refind/3/checkpoint-1500" \
#                  "outputs/lb_refind/4/checkpoint-1500" \
#                  "outputs/lb_refind/5/checkpoint-1500" \
#                  "outputs/lb_refind/6/checkpoint-1500" \
#                  "outputs/lb_refind/7/checkpoint-1500" \
#                  "outputs/lb_refind/8/checkpoint-1500" \
#                  "outputs/lb_refind/9/checkpoint-1500" \
#                  "outputs/lb_refind/10/checkpoint-1500" \
#                  "outputs/lb_refind/-1/checkpoint-1500" \
#     --data_file data/refind/test.json \
#     --device cuda

# python src/tsne.py \
#     --checkpoints "outputs/lb_refind/-3/checkpoint-1500" \
#                  "outputs/lb_refind/-2/checkpoint-1500" \
#                  "outputs/lb_refind/0/checkpoint-1500" \
#                  "outputs/lb_refind/1/checkpoint-1500" \
#                  "outputs/lb_refind/2/checkpoint-1500" \
#                  "outputs/lb_refind/3/checkpoint-1500" \
#                  "outputs/lb_refind/4/checkpoint-1500" \
#                  "outputs/lb_refind/5/checkpoint-1500" \
#                  "outputs/lb_refind/6/checkpoint-1500" \
#                  "outputs/lb_refind/7/checkpoint-1500" \
#                  "outputs/lb_refind/8/checkpoint-1500" \
#                  "outputs/lb_refind/9/checkpoint-1500" \
#                  "outputs/lb_refind/10/checkpoint-1500" \
#                  "outputs/lb_refind/-1/checkpoint-1500" \
#     --data_file data/refind/test_cf.json \
#     --device cuda



group,vib_layer,filepath,f1_CF_Test,f1_Test,precision_CF_Test,precision_Test,recall_CF_Test,recall_Test
lb_refind,-1,outputs/lb_refind/-1/final_results-64--1.txt,72.2789,72.6085,72.095,71.7917,72.4638,73.4442
lb_refind,-2,outputs/lb_refind/-2/final_results-64--2.txt,72.092,74.2746,74.9081,75.4842,69.48,73.1032
lb_refind,-3,outputs/lb_refind/-3/final_results-64-None.txt,71.3247,73.5608,73.8476,73.5922,68.9685,73.5294
lb_refind,0,outputs/lb_refind/0/final_results-64-0.txt,71.3247,73.5608,73.8476,73.5922,68.9685,73.5294
lb_refind,1,outputs/lb_refind/1/final_results-64-1.txt,72.2619,72.7693,71.157,71.7005,73.4015,73.8704
lb_refind,10,outputs/lb_refind/10/final_results-64-10.txt,71.8857,72.4324,71.9471,70.6981,71.8244,74.254
lb_refind,2,outputs/lb_refind/2/final_results-64-2.txt,72.9958,73.3039,72.264,72.2567,73.7425,74.3819
lb_refind,3,outputs/lb_refind/3/final_results-64-3.txt,72.6661,73.3473,71.7012,72.0214,73.6573,74.7229
lb_refind,4,outputs/lb_refind/4/final_results-64-4.txt,73.3418,73.1259,72.6968,71.6742,73.9983,74.6377
lb_refind,5,outputs/lb_refind/5/final_results-64-5.txt,72.8455,72.5057,71.688,70.3407,74.0409,74.8082
lb_refind,6,outputs/lb_refind/6/final_results-64-6.txt,73.0304,72.2552,71.4519,69.4303,74.6803,75.3197
lb_refind,7,outputs/lb_refind/7/final_results-64-7.txt,72.7081,73.1124,72.0687,71.9653,73.3589,74.2967
lb_refind,8,outputs/lb_refind/8/final_results-64-8.txt,73.1571,73.0438,73.7755,71.9124,72.549,74.2114
lb_refind,9,outputs/lb_refind/9/final_results-64-9.txt,71.1054,73.0358,71.1966,71.8171,71.0145,74.2967
lb_retacred,-1,outputs/lb_retacred/-1/final_results-64--1.txt,34.1276,65.2547,30.8382,67.3577,38.2026,63.279
lb_retacred,-2,outputs/lb_retacred/-2/final_results-64--2.txt,55.4452,87.0177,48.9187,87.5874,63.9814,86.4554
lb_retacred,-3,outputs/lb_retacred/-3/final_results-64-None.txt,56.4417,88.2484,50.3367,88.9548,64.232,87.5531
lb_retacred,0,outputs/lb_retacred/0/final_results-64-0.txt,50.0,77.8665,41.4255,77.1692,63.0505,78.5765
lb_retacred,1,outputs/lb_retacred/1/final_results-64-1.txt,38.6936,68.3485,32.7385,69.5836,47.2968,67.1565
lb_retacred,10,outputs/lb_retacred/10/final_results-64-10.txt,54.3419,85.9655,49.104,87.1543,60.8306,84.8088
lb_retacred,2,outputs/lb_retacred/2/final_results-64-2.txt,43.523,74.0774,36.9261,74.9637,52.9896,73.2118
lb_retacred,3,outputs/lb_retacred/3/final_results-64-3.txt,43.3201,73.9016,36.6865,73.3601,52.8822,74.4511
lb_retacred,4,outputs/lb_retacred/4/final_results-64-4.txt,45.5361,76.4085,39.5822,76.5783,53.5983,76.2394
lb_retacred,5,outputs/lb_retacred/5/final_results-64-5.txt,47.5749,77.6455,39.1667,74.229,60.58,81.3916
lb_retacred,6,outputs/lb_retacred/6/final_results-64-6.txt,49.4834,80.0604,40.5585,76.1008,63.4443,84.4547
lb_retacred,7,outputs/lb_retacred/7/final_results-64-7.txt,53.3644,84.5363,47.1444,84.6714,61.4751,84.4016
lb_retacred,8,outputs/lb_retacred/8/final_results-64-8.txt,54.468,85.3554,48.7151,85.891,61.7615,84.8265
lb_retacred,9,outputs/lb_retacred/9/final_results-64-9.txt,55.2674,86.0079,50.4886,87.6194,61.0455,84.4547
lb_tacred,-1,outputs/lb_tacred/-1/final_results-64--1.txt,0.0,0.0,100.0,100.0,0.0,0.0
lb_tacred,-2,outputs/lb_tacred/-2/final_results-64--2.txt,59.1451,64.4882,62.3254,66.4751,56.2736,62.6165
lb_tacred,-3,outputs/lb_tacred/-3/final_results-64-None.txt,61.8648,69.1538,67.6394,69.7461,56.9987,68.5714
lb_tacred,0,outputs/lb_tacred/0/final_results-64-0.txt,61.8648,69.1538,67.6394,69.7461,56.9987,68.5714
lb_tacred,1,outputs/lb_tacred/1/final_results-64-1.txt,0.0,0.0,100.0,100.0,0.0,0.0
lb_tacred,10,outputs/lb_tacred/10/final_results-64-10.txt,26.0744,28.4693,54.6828,55.3913,17.1185,19.1579
lb_tacred,2,outputs/lb_tacred/2/final_results-64-2.txt,35.1197,35.9796,44.2105,45.593,29.1299,29.7143
lb_tacred,3,outputs/lb_tacred/3/final_results-64-3.txt,17.8853,18.9795,53.9683,52.5956,10.7188,11.5789
lb_tacred,4,outputs/lb_tacred/4/final_results-64-4.txt,18.7583,20.1245,60.5852,58.5507,11.0971,12.1504
lb_tacred,5,outputs/lb_tacred/5/final_results-64-5.txt,44.7562,47.9961,48.9338,52.8773,41.2358,43.9398
lb_tacred,6,outputs/lb_tacred/6/final_results-64-6.txt,20.7812,21.7178,50.4866,48.9451,13.0832,13.9549
lb_tacred,7,outputs/lb_tacred/7/final_results-64-7.txt,39.9542,42.411,50.6783,54.0,32.976,34.9173
lb_tacred,8,outputs/lb_tacred/8/final_results-64-8.txt,41.2352,45.4434,49.8658,54.5992,35.1513,38.9173
lb_tacred,9,outputs/lb_tacred/9/final_results-64-9.txt,25.264,26.4283,35.5023,37.3421,19.6091,20.4511
lb_tacrev,-1,outputs/lb_tacrev/-1/final_results-64--1.txt,0.0,0.0,100.0,100.0,0.0,0.0
lb_tacrev,-2,outputs/lb_tacrev/-2/final_results-64--2.txt,59.1451,73.6051,62.3254,73.4994,56.2736,73.7112
lb_tacrev,-3,outputs/lb_tacrev/-3/final_results-64-None.txt,61.2417,78.5304,69.0119,78.3551,55.0441,78.7064
lb_tacrev,0,outputs/lb_tacrev/0/final_results-64-0.txt,61.2417,78.5304,69.0119,78.3551,55.0441,78.7064
lb_tacrev,1,outputs/lb_tacrev/1/final_results-64-1.txt,0.0,0.0,100.0,100.0,0.0,0.0
lb_tacrev,10,outputs/lb_tacrev/10/final_results-64-10.txt,26.0744,31.6873,54.6828,58.8696,17.1185,21.6779
lb_tacrev,2,outputs/lb_tacrev/2/final_results-64-2.txt,35.1197,40.4159,44.2105,49.3309,29.1299,34.2299
lb_tacrev,3,outputs/lb_tacrev/3/final_results-64-3.txt,17.8853,21.3748,53.9683,56.2842,10.7188,13.1924
lb_tacrev,4,outputs/lb_tacrev/4/final_results-64-4.txt,18.7583,22.6069,60.5852,62.4638,11.0971,13.8008
lb_tacrev,5,outputs/lb_tacrev/5/final_results-64-5.txt,44.6383,53.1555,48.785,57.0485,41.1412,49.7598
lb_tacrev,6,outputs/lb_tacrev/6/final_results-64-6.txt,20.7812,23.9745,50.4866,51.4768,13.0832,15.626
lb_tacrev,7,outputs/lb_tacrev/7/final_results-64-7.txt,39.9542,47.4872,50.6783,58.2326,32.976,40.0897
lb_tacrev,8,outputs/lb_tacrev/8/final_results-64-8.txt,41.2352,50.8283,49.8658,58.903,35.1513,44.7006
lb_tacrev,9,outputs/lb_tacrev/9/final_results-64-9.txt,25.264,29.1667,35.5023,39.5936,19.6091,23.0868
ll_retacred,-1,outputs/ll_retacred/-1/final_results-64--1.txt,52.1617,83.7625,46.3659,83.7255,59.6133,83.7996
ll_retacred,-2,outputs/ll_retacred/-2/final_results-64--2.txt,58.7559,90.2037,50.8915,89.8472,69.4952,90.563
ll_retacred,-3,outputs/ll_retacred/-3/final_results-64-None.txt,57.9885,90.7302,52.1913,90.8268,65.2345,90.6339
ll_retacred,0,outputs/ll_retacred/0/final_results-64-0.txt,55.8472,87.01,47.1095,87.5717,68.5643,86.4554
ll_retacred,1,outputs/ll_retacred/1/final_results-64-1.txt,53.9524,83.8435,44.7966,84.0299,67.8124,83.6579
ll_retacred,10,outputs/ll_retacred/10/final_results-64-10.txt,57.4543,89.0289,48.811,87.9972,69.8174,90.085
ll_retacred,2,outputs/ll_retacred/2/final_results-64-2.txt,52.9328,82.7458,44.0791,83.0037,66.237,82.4894
ll_retacred,3,outputs/ll_retacred/3/final_results-64-3.txt,55.6914,85.3331,46.9218,85.7373,68.4927,84.9327
ll_retacred,4,outputs/ll_retacred/4/final_results-64-4.txt,55.3144,86.0748,45.4232,84.6457,70.7125,87.5531
ll_retacred,5,outputs/ll_retacred/5/final_results-64-5.txt,55.6976,86.517,46.172,84.7749,70.1754,88.3322
ll_retacred,7,outputs/ll_retacred/7/final_results-64-7.txt,57.1179,87.9958,48.1236,87.4192,70.247,88.58
ll_retacred,8,outputs/ll_retacred/8/final_results-64-8.txt,57.4338,88.5082,48.3705,87.8444,70.6767,89.182
ll_tacred,-1,outputs/ll_tacred/-1/final_results-64--1.txt,0.0,0.0,100.0,100.0,0.0,0.0
ll_tacred,-2,outputs/ll_tacred/-2/final_results-64--2.txt,64.0323,67.3101,68.6126,71.0184,60.0252,63.9699
rb_refind,-1,outputs/rb_refind/-1/final_results-64--1.txt,72.1381,72.9467,72.1535,72.4558,72.1228,73.4442
rb_refind,-2,outputs/rb_refind/-2/final_results-64--2.txt,72.8747,73.8719,74.9775,74.4801,70.8866,73.2737
rb_refind,-3,outputs/rb_refind/-3/final_results-64-None.txt,73.1202,73.5736,75.7423,74.0501,70.6735,73.1032
rb_refind,0,outputs/rb_refind/0/final_results-64-0.txt,72.6288,73.2419,73.8116,73.6842,71.4834,72.8048
rb_refind,1,outputs/rb_refind/1/final_results-64-1.txt,71.8957,72.4788,69.8713,70.4793,74.0409,74.5951
rb_refind,10,outputs/rb_refind/10/final_results-64-10.txt,71.5875,72.6269,71.8643,72.3656,71.3129,72.89
rb_refind,2,outputs/rb_refind/2/final_results-64-2.txt,72.7311,72.5321,72.0318,71.6424,73.4442,73.4442
rb_refind,3,outputs/rb_refind/3/final_results-64-3.txt,72.6133,73.0409,71.2003,71.2779,74.0835,74.8934
rb_refind,4,outputs/rb_refind/4/final_results-64-4.txt,72.8083,73.4917,72.5159,72.501,73.1032,74.5098
rb_refind,5,outputs/rb_refind/5/final_results-64-5.txt,72.4858,73.2064,71.7146,71.8686,73.2737,74.5951
rb_refind,6,outputs/rb_refind/6/final_results-64-6.txt,72.9012,73.0826,72.7003,72.068,73.1032,74.1262
rb_refind,7,outputs/rb_refind/7/final_results-64-7.txt,73.5661,73.1738,73.0559,72.0844,74.0835,74.2967
rb_refind,8,outputs/rb_refind/8/final_results-64-8.txt,72.9597,72.6812,72.9442,72.057,72.9753,73.3163
rb_refind,9,outputs/rb_refind/9/final_results-64-9.txt,71.4013,73.0339,71.0249,72.257,71.7818,73.8278
rb_retacred,-1,outputs/rb_retacred/-1/final_results-64--1.txt,34.0913,66.0449,30.3292,66.2627,38.9187,65.8286
rb_retacred,-2,outputs/rb_retacred/-2/final_results-64--2.txt,56.0293,88.2046,48.88,88.1656,65.6284,88.2436
rb_retacred,-3,outputs/rb_retacred/-3/final_results-64-None.txt,56.5238,87.7805,50.3356,88.3154,64.4468,87.2521
rb_retacred,0,outputs/rb_retacred/0/final_results-64-0.txt,51.1774,79.6538,42.5486,79.4574,64.1962,79.8513
rb_retacred,1,outputs/rb_retacred/1/final_results-64-1.txt,49.1035,78.4113,40.5361,77.0058,62.2628,79.869
rb_retacred,10,outputs/rb_retacred/10/final_results-64-10.txt,54.2915,85.5567,48.324,85.9543,61.9406,85.1629
rb_retacred,2,outputs/rb_retacred/2/final_results-64-2.txt,44.4093,73.6215,37.0948,72.912,55.3169,74.3449
rb_retacred,3,outputs/rb_retacred/3/final_results-64-3.txt,45.3522,76.082,37.0572,73.1883,58.4318,79.2139
rb_retacred,4,outputs/rb_retacred/4/final_results-64-4.txt,45.2559,76.3757,38.3259,75.2535,55.2453,77.5319
rb_retacred,5,outputs/rb_retacred/5/final_results-64-5.txt,49.8334,80.8102,41.8491,79.0517,61.5825,82.6487
rb_retacred,6,outputs/rb_retacred/6/final_results-64-6.txt,51.1151,82.2981,44.1322,81.7943,60.7232,82.8081
rb_retacred,7,outputs/rb_retacred/7/final_results-64-7.txt,51.9096,83.1905,45.2705,83.0072,60.8306,83.3746
rb_retacred,8,outputs/rb_retacred/8/final_results-64-8.txt,50.5518,81.8952,44.1973,81.6215,59.0405,82.1707
rb_retacred,9,outputs/rb_retacred/9/final_results-64-9.txt,53.9948,84.8248,49.7436,87.0205,59.0405,82.7373
rb_tacred,-1,outputs/rb_tacred/-1/final_results-64--1.txt,15.4318,15.8046,40.7008,38.7779,9.5208,9.9248
rb_tacred,-2,outputs/rb_tacred/-2/final_results-64--2.txt,62.2454,67.1885,66.775,68.5992,58.2913,65.8346
rb_tacred,-3,outputs/rb_tacred/-3/final_results-64-None.txt,60.8476,68.2826,65.2127,67.1316,57.0303,69.4737
rb_tacred,0,outputs/rb_tacred/0/final_results-64-0.txt,24.0806,26.1162,39.3983,38.0541,17.3392,19.8797
rb_tacred,1,outputs/rb_tacred/1/final_results-64-1.txt,39.724,41.6994,48.6301,51.2956,33.575,35.1278
rb_tacred,10,outputs/rb_tacred/10/final_results-64-10.txt,38.0896,42.1907,51.2807,54.5281,30.2963,34.406
rb_tacred,2,outputs/rb_tacred/2/final_results-64-2.txt,42.7999,46.2185,45.5224,49.9476,40.3846,43.0075
rb_tacred,3,outputs/rb_tacred/3/final_results-64-3.txt,39.5652,40.9326,46.5077,48.073,34.4262,35.6391
rb_tacred,4,outputs/rb_tacred/4/final_results-64-4.txt,19.2686,20.0,48.1715,47.1508,12.0429,12.6917
rb_tacred,5,outputs/rb_tacred/5/final_results-64-5.txt,44.4849,47.2493,46.8597,49.4896,42.3392,45.203
rb_tacred,6,outputs/rb_tacred/6/final_results-64-6.txt,43.1448,45.5777,44.1186,46.1467,42.2131,45.0226
rb_tacred,7,outputs/rb_tacred/7/final_results-64-7.txt,43.1734,45.7864,44.2824,46.3219,42.1185,45.2632
rb_tacred,8,outputs/rb_tacred/8/final_results-64-8.txt,42.7972,45.449,43.8742,46.1682,41.7718,44.7519
rb_tacred,9,outputs/rb_tacred/9/final_results-64-9.txt,43.0855,48.1316,47.9876,51.3109,39.0921,45.3233
rb_tacrev,-1,outputs/rb_tacrev/-1/final_results-64--1.txt,15.4318,17.7151,40.7008,41.3631,9.5208,11.2712
rb_tacrev,-2,outputs/rb_tacrev/-2/final_results-64--2.txt,62.2454,76.8451,66.775,76.0263,58.2913,77.6817
rb_tacrev,-3,outputs/rb_tacrev/-3/final_results-64-None.txt,60.2064,78.2266,67.6228,77.1073,54.256,79.3788
rb_tacrev,0,outputs/rb_tacrev/0/final_results-64-0.txt,24.0806,28.0247,39.3983,39.2055,17.3392,21.806
rb_tacrev,1,outputs/rb_tacrev/1/final_results-64-1.txt,39.724,47.2222,48.6301,55.9947,33.575,40.8261
rb_tacrev,2,outputs/rb_tacrev/2/final_results-64-2.txt,42.7999,51.7541,45.5224,54.1041,40.3846,49.5997
rb_tacrev,3,outputs/rb_tacrev/3/final_results-64-3.txt,39.5652,45.9914,46.5077,52.1298,34.4262,41.1463
rb_tacrev,4,outputs/rb_tacrev/4/final_results-64-4.txt,19.2686,22.2499,48.1715,49.9441,12.0429,14.3132
rb_tacrev,5,outputs/rb_tacrev/5/final_results-64-5.txt,44.2078,52.64,46.0252,52.7671,42.5284,52.5136
rb_tacrev,6,outputs/rb_tacrev/6/final_results-64-6.txt,36.1887,42.1834,45.0658,50.1989,30.2333,36.3753
rb_tacrev,7,outputs/rb_tacrev/7/final_results-64-7.txt,40.055,46.9131,53.0973,59.2955,32.1564,38.8088
rb_tacrev,8,outputs/rb_tacrev/8/final_results-64-8.txt,41.7587,49.5797,63.9974,68.9282,30.9899,38.7128
rb_tacrev,9,outputs/rb_tacrev/9/final_results-64-9.txt,43.0855,53.7954,47.9876,55.4988,39.0921,52.1934
