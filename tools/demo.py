# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import shutil
import os
import datetime
import subprocess
from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--writevid', action='store_true', help='save output as a video (requires ffmpeg)')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Set output dir
    timestamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
    output_dir = os.path.join('./output', timestamp, 'frames')
    output_video_path = os.path.join('./output', timestamp, 'output.mp4')
    # Create dir
    os.makedirs(output_dir, exist_ok=True)

    # Select ROI
    # cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        # init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        # x, y, w, h = init_rect
        x, y, w, h = 300, 110, 165, 250
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            # cv2.imshow('SiamMask', im)
            # key = cv2.waitKey(1)
            # if key > 0:
            #     break
            filepath = os.path.join(output_dir, str(f).zfill(5) + '.jpg')
            cv2.imwrite(filepath, im)

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

    if args.writevid:
        print('Writing output to video', output_video_path)
        # requires ffmpeg
        shell_command = f'ffmpeg -hide_banner -r 20 -f image2 -i {output_dir}/%5d.jpg -crf 4 -vcodec libx264 -pix_fmt yuv420p {output_video_path}'.split(' ')
        subprocess.run(shell_command)
