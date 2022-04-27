import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import seaborn as sns
import utils


parser = argparse.ArgumentParser(description='Plane visualization')
parser.add_argument('--w0', type=int, default=100)
parser.add_argument('--w1', type=int, default=200)
parser.add_argument('--w2', type=int, default=300)
parser.add_argument('--scale', type=str, default='normal')
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--dir', type=str, default='./', metavar='DIR',
                    help='training directory (default: None)')

args = parser.parse_args()


# 저장했던 npz 로드
file = np.load(os.path.join(args.dir, 'plane.npz'))

# plot font 설정 #TODO: 최신 버전으로 수정필요
matplotlib.rc('text', usetex=True)
#matplotlib.rc('text.latex', preamble=[r'\usepackage{sansmath}', r'\sansmath'])
matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})

# x축 y축 tick(눈금) 관련 setting
matplotlib.rc('xtick.major', pad=12)
matplotlib.rc('ytick.major', pad=12)
matplotlib.rc('grid', linewidth=0.8)

# 뒷배경 스타일
sns.set_style('whitegrid')

# contour그릴 때, log스케일로 할 예정 /// 이걸로 안하면 default는 Linear 스케일로 되는 듯 함
# TODO: 아직 정확하게 이해하지 않음
class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)

# plot하는 함수
def plane(grid, values, vmax=None, log_alpha=-5, N=7, cmap='jet_r'):
    '''
    :param grid: 좌표평면
    :param values: tr_res te_res 에 속했던거 정리한 정보들
    :param vmax: 좌표에 해당하는 포인트가 가질 수 있는 정보값의 상한, 최대값 설정
    :param log_alpha: 상한-하한 값이 음수가 안되도록 잡아주면 되는 걸로 예상됨 (아래 log_gamma 참고) ( log_alpha 자체는 음수 )
    :param N: color 몇개로 표시할 지 결정
    :param cmap: contour style
    :return: contour, contourf, colorbar
    '''
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
    else:
        clipped = np.minimum(values, vmax) #상한값 이상은 상한값으로 대체
    log_gamma = (np.log(clipped.max() - clipped.min())- log_alpha) / N #color가 달라지는 값의 간격
    levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1)) # color가 달라지는 값 자체

    levels[0] = clipped.min() # color 표시값 하한값
    levels[-1] = clipped.max() # color 표시값의 상한값
    levels = np.concatenate((levels, [1e10])) # 무조건 제일 큰값 추가 <- 왜 하는지 아직 파악안함

    # 위에서 말한 log scale로 바꾸는 걸로 예상하는 부분
    norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)

    # 그리기 -- 이 때, 처음 3개의 argument는 2d로 동일한 shape
    # 밑그림
    contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                          linewidths=2.5,
                          zorder=1, # layer 순서 두번쨰
                          levels=levels)

    # 색채움
    contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                            levels=levels, # contour level
                            zorder=0,  # layer순서 첫번쨰
                            alpha=0.55) # 투명도

    # colorbar 지정
    # TODO: 아직 정확히 파악 안함
    colorbar = plt.colorbar(format='%.2g')
    labels = list(colorbar.ax.get_yticklabels())
    labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
    colorbar.ax.set_yticklabels(labels)
    return contour, contourf, colorbar

path = os.path.join(args.dir, f'{args.w0}_{args.w1}_{args.w2}')
if not os.path.exists(path):
    os.mkdir(path)

##########################################
#             Train Plot                 #
##########################################
for i in ['tr_nll','tr_loss','tr_acc']:
    plt.figure(figsize=(12.4, 7))

    contour, contourf, colorbar = plane(
        file['grid'],
        file[i], # 이 부분을 다른 정보로 바꿔주면서 error surfacce, nll surface, acc surface로 다르게 그릴 수 있음
        vmax=15.0,      # 단 log_alpha 조절 필요
        log_alpha=-15.0,   # TODO: ERROR중에 Contour levels must be increasing 뜨면 값을 더 낮추면 해결
        N=7
    )

    # 초기 fix_point 불러옴
    bend_coordinates = file['bend_coordinates']

    #curve_coordinates = file['curve_coordinates'] #TODO: 필요없음 --> curve에 관한 모든 값은 Fast geometric ensembling 내용

    # fix_point[0] = 검정
    plt.scatter(bend_coordinates[0, 0], bend_coordinates[0, 1], marker='X', c='k', s=120, zorder=2)

    # fix_point[1] = 빨강
    plt.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1], marker='X', c='r', s=120, zorder=2)

    # fix_point[2] = 파랑
    plt.scatter(bend_coordinates[2, 0], bend_coordinates[2, 1], marker='X', c='b', s=120, zorder=2)

    # plot 설정
    plt.title(f"Train_{i}")
    plt.grid(False)
    plt.margins(0.0)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    colorbar.ax.tick_params(labelsize=18)

    # plot 저장
    plt.savefig(os.path.join(path, f'{args.scale}_seed_{args.random_seed}_train_{i}_{args.w0}_{args.w1}_{args.w2}.jpg'),
                format='jpg', bbox_inches='tight')

    plt.show()

##########################################
#             Test Plot                  #
##########################################
for i in ['te_nll', 'te_loss', 'te_acc']:
    plt.figure(figsize=(12.4, 7))

    contour, contourf, colorbar = plane(
        file['grid'],
        file[i],   # 이 부분을 다른 정보로 바꿔주면서 error surfacce, nll surface, acc surface로 다르게 그릴 수 있음
        vmax=15,          # 단 log_alpha 조절 필요
        log_alpha=-10.0,
        N=7
    )

    bend_coordinates = file['bend_coordinates']

    # fix_point[0] = 검정
    plt.scatter(bend_coordinates[0, 0], bend_coordinates[0, 1], marker='X', c='k', s=120, zorder=2)

    # fix_point[1] = 빨강
    plt.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1], marker='X', c='r', s=120, zorder=2)

    # fix_point[2] = 파랑
    plt.scatter(bend_coordinates[2, 0], bend_coordinates[2, 1], marker='X', c='b', s=120, zorder=2)

    # plot 설정
    plt.title(f"Test_{i}")
    plt.grid(False)
    plt.margins(0.0)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    colorbar.ax.tick_params(labelsize=10)

    # plot 저장
    plt.savefig(os.path.join(path, f'{args.scale}_seed_{args.random_seed}_test_{i}_{args.w0}_{args.w1}_{args.w2}.jpg'),
                format='jpg', bbox_inches='tight')

    plt.show()