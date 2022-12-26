import time
import multiprocessing as mp
import cv2

def image_put(q,index):
    # print("image_put,",str(q),str(index))
    cap = cv2.VideoCapture(index)
    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name):
    # print("image_get,", str(q), str(window_name))
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        print(frame)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_multi_camera():
    camera_ip_l = [0,1,2]
    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]
    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue,  camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, str(camera_ip))))
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def image_collect(queue_list, camera_ip_l):
    import numpy as np

    """show in single opencv-imshow window"""
    window_name = "%s_and_so_no" % camera_ip_l[0]
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        imgs = [q.get() for q in queue_list]
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow(window_name, imgs)
        if cv2.waitKey(1)&0xff==(ord ('q')):
            break

    # """show in multiple opencv-imshow windows"""
    # [cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    #  for window_name in camera_ip_l]
    # while True:
    #     for window_name, q in zip(camera_ip_l, queue_list):
    #         cv2.imshow(window_name, q.get())
    #         cv2.waitKey(1)

def run_multi_camera_in_a_window():
    user_name, user_pwd = "admin", "admin123456"
    camera_ip_l = [
        "172.20.114.196",  # ipv4
        "[fe80::3aaf:29ff:fed3:d260]",  # ipv6
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = [mp.Process(target=image_collect, args=(queues, camera_ip_l))]
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
    for process in processes:
        process.join()

def run():
    run_multi_camera() # with 1 + n threads
    # run_multi_camera_in_a_window()  # with 1 + n threads
    pass

# def main_test():
#     cv2.VideoCapture(index) camera1 = new VideoCapture(0);
#     if (!camera1.isOpened()) {
#         System.out.println("Error! Camera1 can't be opened!");
#         return;
#     }

if __name__ == '__main__':
    run()








