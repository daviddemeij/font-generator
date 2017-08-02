import os
import argparse
import imageio
import numpy as np
parser = argparse.ArgumentParser(description='GIF creator')
parser.add_argument('--experiment_name', type=str, default='default', help='experiment name')
parser.add_argument('--frame_duration', type=float, default=0.1, help='Set frame duration')

def main():
    global args
    args = parser.parse_args()
    experiment_dir = os.path.join("/home/david/training_logs/GAN", args.experiment_name)
    images = []
    for filename in sorted(os.listdir(os.path.join(experiment_dir, "images"))):
        if filename.endswith(".png"):
            print filename
            images.append(imageio.imread(os.path.join(os.path.join(experiment_dir, "images"), filename)))
    # Save them as frames into a gif
    exportname = os.path.join(experiment_dir, "gif.gif")
    kargs = {'duration': args.frame_duration}
    imageio.mimsave(exportname, images, 'GIF', **kargs)

if __name__== "__main__":
    main()

