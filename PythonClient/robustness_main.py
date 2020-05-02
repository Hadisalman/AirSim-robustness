import argparse
import time

from robustness.tools.attacks import PGD
from configs import attack_config

from robustness import Car, Pedestrian, Weather, AdversarialObjects
from IPython import embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo for the airsim-robustness package')
    parser.add_argument('model', metavar='DIR',
                        help='path to pretrained model')
    parser.add_argument('--demo-id', type=int, choices=[0, 1, 2, 3, 4],
                        help='which task of the demo to excute'
                        '0 -> image callback thread'
                        '1 -> test all threads'
                        '2 -> search for 3D advesarial configuration'
                        '3 -> read adv_config.json and run ped recognition'
                        '4 -> pixel pgd attack'
                        )
    parser.add_argument('--img-size', default=224, type=int, metavar='N',
                        help='size of rgb image (assuming equal height and width)')
    parser.add_argument('--resolution-coord-descent', default=10, type=int,
                        help='resolution of coord descent 3D object adv attack')
    parser.add_argument('--num-iter', default=1, type=int,
                        help='number of iterators of coord descent 3D object adv attack')
    parser.add_argument('--adv-config-path', type=str, default='./results.json')

    args = parser.parse_args()

    car = Car(detection_model=args.model)
    ped = Pedestrian()
    weather = Weather()
    adversary = AdversarialObjects('adversary', car, 
                            resolution_coord_descent=args.resolution_coord_descent,
                            num_iter=args.num_iter,
                            adv_config_path=args.adv_config_path)

    # embed()
    if args.demo_id == 0:
        car.detection.display_image_stream()

    if args.demo_id == 1:
        car.detection.run()
        time.sleep(3)
        ped.walk()
        time.sleep(2)
        car.drive()
        weather.start()

    if args.demo_id == 2:
        car.client.simPause(True)

        adversary.adv_config_path = './adv_configs/config_fp_2.json'
        #remove ped from scene
        ped.hide()
        adversary.attack()        
 
        adversary.adv_config_path = './adv_configs/config_fn_4.json'
        adversary.attack()        

    elif args.demo_id == 3:
        car.client.simPause(True)

        # adversary.update_env_from_config(path='./adv_configs/config_fp.json')
        adversary.update_env_from_config(path='./adv_configs/config_fp_2.json')
        
        # adversary.update_env_from_config(path='./adv_configs/config_fn.json')
        # adversary.update_env_from_config(path='./adv_configs/config_fn_2.json')
        # adversary.update_env_from_config(path='./adv_configs/config_fn_3.json')
        car.detection.run()
        # car.drive()

    elif args.demo_id == 4:
        car.detection.setup_attack(attack_config)
        car.detection.run(with_attack=True)
        time.sleep(3)
        ped.walk()
        time.sleep(2)
        car.drive()

    embed()
    
    adversary.reset()
    car.reset()
    weather.reset()
    ped.reset()
