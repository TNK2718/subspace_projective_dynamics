# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Plots a cloth trajectory rollout."""
import os
import pickle
import pathlib
from absl import app
from absl import flags
from matplotlib import animation
import matplotlib.pyplot as plt
import math
import pd_model
import geometry_init

#
dataset_name = 'flag'
rollout_name = '1'

#
root_dir = pathlib.Path(__file__).parent.resolve()
output_dir = os.path.join(root_dir, 'output', dataset_name)
rollout_dir = os.path.join(output_dir, 'rollout')
# rollout_path = os.path.join(rollout_dir, 'fullspace_rollout.pkl')
data_dir = os.path.join(root_dir, 'data', dataset_name)

#
fullspace_data_path = os.path.join(data_dir, 'fullspace_traj.pkl')
pca_base_path = os.path.join(data_dir, 'pca_base.pkl')


def main(unused_argv):
    print('fullspace rollout starting...')

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111, projection='3d')
    skip = 10
    num_steps = 100
    num_frames = num_steps

    # Setup solvers
    models = []

    # flag
    res_w = 50
    res_h = 30
    len_w = 0.5
    len_h = 0.3
    models.append(geometry_init.generate_plane(res_w, res_h, len_w, len_h))

    def animate(num):
        step = (num * skip) % num_steps

        ax.cla()

        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([-1.0, 1.0])

        for model in models:
            vert = model.rendering_verts
            faces = model.faces
            ax.plot_trisurf(vert[:, 0], vert[:, 1],
                            faces, vert[:, 2], shade=True)

        ax.set_title('Step %d' % (step))
        print('Step: %d' % (step))

        # advance time
        for model in models:
            model.simulate()

        return fig,

    # ani = animation.FuncAnimation(fig, animate, frames=math.floor(num_frames * 0.1), interval=100)
    # ani = animation.FuncAnimation(fig, animate, interval=100)
    # ani = animation.FuncAnimation(
    #     fig, animate, frames=num_frames * 100, interval=50)
    ani = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=50)

    ani.save(os.path.join(rollout_dir, 'fullspace_traj.mp4'), writer="ffmpeg")
    plt.show(block=True)


if __name__ == '__main__':
    app.run(main)
