"""Semi-reduced projective dynamics in PCA-subspace"""
import time
import os
import pickle
import pathlib
from pathlib import Path
from tracemalloc import start
from absl import app
from absl import flags
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import sub_pd_model
import geometry_init
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#
dataset_name = 'flag_test'
rollout_name = '1'
geometry_func = geometry_init.generate_plane

#
root_dir = pathlib.Path(__file__).parent.resolve()
output_dir = os.path.join(root_dir, 'output', dataset_name)
rollout_dir = os.path.join(output_dir, 'rollout')
# rollout_path = os.path.join(rollout_dir, 'fullspace_rollout.pkl')
data_dir = os.path.join(root_dir, 'data', dataset_name)

#
# fullspace_data_path = os.path.join(data_dir, 'fullspace_traj.npz')
pca_base_path = os.path.join(data_dir, 'pca_base.npz')

'''Paramters'''
pca_dim = 10
# geometry: flag
res_w = 5
res_h = 3
len_w = 1.5
len_h = 0.9
# Animation setting
skip = 1
num_steps = 500
num_frames = num_steps

def load_base(dir):
    ndarr = np.load(os.path.join(dir, 'pca_base.npz'))
    base = ndarr['base']
    center = ndarr['mean']
    return (base, center)

def prepare_files_and_directories():
    # make all the necessary directories
    Path(rollout_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

def main(unused_argv):
    #
    prepare_files_and_directories()
    
    #
    print('Subspace rollout starting...')

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111, projection='3d')

    # Setup solvers
    models = []

    # Load base matrix
    sub_args = load_base(data_dir)

    # models.append(geometry_init.generate_plane(res_w, res_h, len_w, len_h))
    models.append(sub_pd_model.SubPDModel(*geometry_func(res_w, res_h, len_w, len_h), *sub_args))
    start = time.time()

    # Trajectory of a cloth
    sub_s_0_traj = np.zeros((num_frames, 3 * models[0].r))
    sub_p_term_traj = np.zeros((num_frames, 3 * models[0].r))

    def animate(num):
        step = (num * skip) % num_steps

        ax.cla()

        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([-1.0, 1.0])

        for model in models:
            vert = np.copy(model.rendering_verts)
            faces = model.rendering_faces
            ax.plot_trisurf(vert[:, 0], vert[:, 1],
                            faces, vert[:, 2], shade=True)

        ax.set_title('Step %d' % (step))
        print('Step: %d' % (step))
        print(time.time() - start)
        # print(models[0].rendering_verts[0,:])

        # save positions
        sub_s_0_traj[num, :] = np.copy(model.sub_s_0_snapshot)
        sub_p_term_traj[num, :] = np.copy(model.p_term_snapshot)

        # advance time
        for _ in range(skip):
            for model in models:
                model.simulate()
        return fig,

    # ani = animation.FuncAnimation(fig, animate, frames=math.floor(num_frames * 0.1), interval=100)
    # ani = animation.FuncAnimation(fig, animate, interval=100)
    # ani = animation.FuncAnimation(
    #     fig, animate, frames=num_frames * 100, interval=50)
    ani = animation.FuncAnimation(
        fig, animate, frames=num_frames)

    ani.save(os.path.join(rollout_dir, 'fullspace_traj.mp4'), writer="ffmpeg")
    plt.show(block=True)

    '''Save trajectory'''
    np.savez(os.path.join(data_dir, 'subspace_traj.npz'), s_0=sub_s_0_traj, p_term=sub_p_term_traj)

if __name__ == '__main__':
    app.run(main)
