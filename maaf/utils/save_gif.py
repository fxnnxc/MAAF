from matplotlib import animation
import matplotlib.pyplot as plt
import os 


def save_frames_as_gif(frames, path, title="", fps=30):
    # sudo apt-get install imagemagick
    # sudo apt-get install imagemagick

    print(len(frames))
    fig = plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    patch = plt.imshow(frames[0])
    # plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval=50)
    # plt.show()
    writergif = animation.PillowWriter(fps=fps) 
    anim.save(path+".gif", writer=writergif)
    # anim.save(path+".gif", writer='imagemagick', fps=60)