import torch
import torch.nn as nn
import torch
import numpy as np
import tabulate
import utils
import os
import argparse

parser = argparse.ArgumentParser(description='Plane visualization')
parser.add_argument('--w0', type=int, default=300)
parser.add_argument('--w1', type=int, default=320)
parser.add_argument('--w2', type=int, default=340)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--method', type=str, default='IRM')
parser.add_argument('--margin', type=float, default=2)
args = parser.parse_args()

utils.seed_everything(args.random_seed)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def torch_numpy_flat(v_list):
    '''
    :param v_list:
    :return: 모델의 모든 파라미터를 하나의 벡터로 flatten
    '''
    return torch.concat([torch.flatten(v) for v in v_list]).detach().cpu().numpy()

def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

# main 함수에서 저장한 3개의 서로 다른 iteration의 모델 파라미터
v1 = torch.load(f'./{args.method}/param_{args.w0}.pt')
v2 = torch.load(f'./{args.method}/param_{args.w1}.pt')
v3 = torch.load(f'./{args.method}/param_{args.w2}.pt')

# 각 iteration의 파라미터를 flatten 시킴.
fix_points = []
for i in [v1,v2,v3]:
    fix_points.append(torch_numpy_flat(i)) # IRM 166657

# 2-d flatten surface에서 unit vector 역할을 할 u(x축), v(y축) 단위벡터 생성
unit_vector = []
u = fix_points[2] - fix_points[0]
dx = np.linalg.norm(u)
u /= dx
unit_vector.append(u)

v = fix_points[1] - fix_points[0]
v -= np.dot(u, v) * u
dy = np.linalg.norm(v)
v /= dy
unit_vector.append(v)

# 3개의 iteration을 각각 좌표 평면의 scale로 표시
# 이 때, 원점은 항상 fix_points[0]
bend_coordinates = np.stack([get_xy(p, fix_points[0], unit_vector[0], unit_vector[1]) for p in fix_points])

# TODO: loss surface만 표시하기엔 쓸데없는 부분
#ts = np.linspace(0.0, 1.0, 31) # args.curve_points = 61
#curve_coordinate ????/  ## Fast Geometric Ensembling내용 (필요없음)

# loss 구할 부분은 총 21x21개
G = 21 # args.grid_points
margin = args.margin
alphas = np.linspace(0.0 - margin, 1.0 + margin, G)
betas = np.linspace(0.0 - margin, 1.0 + margin, G)

# loss 말고 acc, nll, err로도 2d로 표현하기 위해
tr_loss, tr_nll, tr_acc, tr_err = np.zeros((G,G)), np.zeros((G,G)), np.zeros((G,G)), np.zeros((G,G))
te_loss, te_nll, te_acc, te_err = np.zeros((G,G)), np.zeros((G,G)), np.zeros((G,G)), np.zeros((G,G))

grid = np.zeros((G,G,2))

# 학습 시 사용했던 모델과 같은 모델
base_model = utils.MLPBase()
base_model.cuda()

# grid마다 값을 구해 저장
columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        p = fix_points[0] + dx*alpha * u + dy*beta * v  # 위치벡터
        offset = 0

        # flatten시켰던 파라미터를 다시 base model에 맞게 layer별로 할당
        for parameter in base_model.parameters():
            size = np.prod(parameter.size())
            value = p[offset:offset+size].reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        # TODO: 현재는 mlp모델만 구현된 상태 BN쓰는 모델은 아직 수정 안함
        # utils.update_bn(loaders['train'], base_model)

        # 평가 (train, test) (train update는 안함)
        tr_res, te_res = utils.evaluation(base_model) # dict()

        tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res['accuracy']
        te_loss_v, te_nll_v, te_acc_v = te_res['loss'], te_res['nll'], te_res['accuracy']

        c = get_xy(p, fix_points[0], u, v)   # 쓸데없음

        # grid 위치 지정
        grid[i, j] = [alpha * dx, beta * dy]

        # train information
        tr_loss[i, j] = tr_loss_v
        tr_nll[i, j] = tr_nll_v
        tr_acc[i, j] = tr_acc_v
        tr_err[i, j] = 100.0 - tr_acc[i, j]*100

        # test information
        te_loss[i, j] = te_loss_v
        te_nll[i, j] = te_nll_v
        te_acc[i, j] = te_acc_v
        te_err[i, j] = 100.0 - te_acc[i, j]*100

        values = [
            grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
            te_nll[i, j], te_err[i, j]
        ]

        # terminal에 표시할 정보들
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if j == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

# npz로 저장해서 plot.py에서 불러와서 사용할 예정 --> plot.py ㄱㄱ
np.savez(
    './plane.npz',
    bend_coordinates=bend_coordinates,
    alphas=alphas,
    betas=betas,
    grid=grid,
    tr_loss=tr_loss,
    tr_acc=tr_acc,
    tr_nll=tr_nll,
    tr_err=tr_err,
    te_loss=te_loss,
    te_acc=te_acc,
    te_nll=te_nll,
    te_err=te_err
)

