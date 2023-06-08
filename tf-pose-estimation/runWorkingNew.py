import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--bgImage', type=str, default='./images/white.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    import os

    imgPath = "/home/abhiram/Documents/poseDetection/tf-pose-estimation/DATASET/TEST/"



    for imgDir in os.listdir(imgPath):
        for img in os.listdir(imgPath + imgDir + "/"):

            white = common.read_imgfile(args.bgImage, None, None)

            if white is None:
                logger.error('bgImage cannot be loaded')
                sys.exit(-1)

            image = common.read_imgfile((imgPath + imgDir + "/" + img), None, None)

            t = time.time()
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            if len(humans) < 1:
                print(img, " failed")
                continue

            humans = humans[:1]
            elapsed = time.time() - t

            '''
            for i in range(len(humans[0].body_parts.keys())):
                print(str(humans[0].body_parts[i]))
                part = str(humans[0].body_parts[i])
                x.append(float(part[13:16]))
                y.append(float(part[19:22]))
                print(x[i], y[i])

            print(x)
            print(y)
            '''

            arrayPath = "/home/abhiram/Documents/poseDetection/preProcessArray/" + imgDir + "/" + img[0:8] + ".npy"
            cvPath = "/home/abhiram/Documents/poseDetection/preProcessedImages/" + imgDir + "/" + img[0:8] + ".png"
            binPath = "/home/abhiram/Documents/poseDetection/preProcessBinary/" + imgDir + "/" + img[0:8] + ".png"

            print("dir: ", imgDir, " | img: ", img)

            #logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

            import numpy as np

            np.save(arrayPath, humans[0].body_parts)
            '''
            
            try:
                
            except Exception as e:
                logger.warning('no pose found for ', img)
                continue

            # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            '''
            image = TfPoseEstimator.draw_humans(white, humans, imgcopy=False)


            try:
                import matplotlib.pyplot as plt

                fig = plt.figure(frameon=False)

                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)

                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                fig.savefig(cvPath)

                #import cv2

                
                

                originalImage = cv2.imread(cvPath)
                grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

                (thresh, bWImage) = cv2.threshold(grayImage, 235, 255, cv2.THRESH_BINARY)
                cv2.imwrite(binPath, bWImage)

                '''
                
                
                import cv2

                originalImage = cv2.imread(cvPath)
                grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

                (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
                '''


                #plt.show()
            except Exception as e:
                logger.warning('matplitlib error, %s' % e)
                #cv2.imshow('result', image)
                #cv2.waitKey()

    


    '''
    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    white = common.read_imgfile(args.bgImage, None, None)

    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    if white is None:
        logger.error('bgImage cannot be loaded')
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)


    humans = humans[:1]
    elapsed = time.time() - t


    x = []
    y = []

    for i in range(len(humans[0].body_parts.keys())):
        print(str(humans[0].body_parts[i]))
        part = str(humans[0].body_parts[i])
        x.append(float(part[13:16]))
        y.append(float(part[19:22]))
        print(x[i], y[i])

    print(x)
    print(y)

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    image = TfPoseEstimator.draw_humans(white, humans, imgcopy=False)

    print(type(image))

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()


        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.savefig
        print(type(image))
        print(type(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        

        plt.show()
    except Exception as e:
        logger.warning('matplitlib error, %s' % e)
        cv2.imshow('result', image)
        cv2.waitKey()
    '''
