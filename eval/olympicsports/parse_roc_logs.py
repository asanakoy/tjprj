import glob
from os.path import join
import re
from collections import defaultdict

if __name__ == '__main__':
    dir_path = '/export/home/asanakoy/workspace/OlympicSports/cnn/ft_alexnet_joint_categories_imagenet_fixall500_200initbatches'

    ALL_CATEGORIES = [
        'basketball_layup',
        'bowling',
        'clean_and_jerk',
        'discus_throw',
        'diving_platform_10m',
        'diving_springboard_3m',
        'hammer_throw',
        'high_jump',
        'javelin_throw',
        'long_jump',
        'pole_vault',
        'shot_put',
        'snatch',
        'tennis_serve',
        'triple_jump',
        'vault']

    LAYERS = ['maxpool5', 'fc6', 'fc7', 'fc8']
    specific_iter = 445004
    results = defaultdict(dict)
    for cat in ALL_CATEGORIES:
        print cat
        for layer_name in LAYERS:
            if specific_iter is None:
                filenames = sorted(glob.glob1(join(dir_path, cat), layer_name + '_*'))
            else:
                filenames = sorted(glob.glob1(join(dir_path, cat), layer_name + '_*_iter-{}'.format(specific_iter)))
            print filenames
            if len(filenames) != 0:
                matches = re.match(layer_name + r'_((?:[01][.])?[0-9]+)_iter-\d+', filenames[-1])
                # matches = re.match(r'((?:[01][.])?[0-9]+)_' + layer_name + r'_checkpoint-\d+', filenames[-1])
            else:
                matches = None
            if matches is not None:
                auc = float(matches.groups()[0])
                print cat, layer_name, auc
                results[cat][layer_name] = auc
            else:
                print cat, layer_name, '---'

        print '=========', cat, '==========='
        print sorted(results[cat].items())
        print '============================='

    for layer_name in LAYERS:
        print '=========', layer_name, '==========='
        for cat in ALL_CATEGORIES:
            if cat not in results or layer_name not in results[cat]:
                print cat
            else:
                print '{}\t {:.3f}'.format(cat, results[cat][layer_name])
