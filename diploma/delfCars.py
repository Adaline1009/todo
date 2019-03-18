import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform


def resize_image(srcfile, destfile, new_width=256, new_height=256):
    pil_image = Image.open(srcfile)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save(destfile, format='JPEG', quality=90)
    return destfile


def resize_images_folder(srcfolder, destfolder='cars/resized', new_width=256, new_height=256):
    os.makedirs(destfolder, exist_ok=True)
    for srcfile in glob.iglob(os.path.join('cars/db_images', '*.[Jj][Pp][Gg]')):
        src_basename = os.path.basename(srcfile)
        destfile = os.path.join(destfolder, src_basename)
        resize_image(srcfile, destfile, new_width, new_height)
    return destfolder


def get_resized_db_image_paths(destfolder='cars/resized'):
    return sorted(list(glob.iglob(os.path.join(destfolder, '*.[Jj][Pp][Gg]'))))


def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)


def compute_locations_and_descriptors(image_path):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.FATAL)

    m = hub.Module('https://tfhub.dev/google/delf/1')

    # The module operates on a single image at a time, so define a placeholder to
    # feed an arbitrary image in.
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)

    image_tf = image_input_fn([image_path])

    with tf.train.MonitoredSession() as sess:
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % image_path)
        return sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})


def preprocess_query_image(imagepath, query_image):
    query_temp_folder_name = 'query_temp_folder'
    query_temp_folder = os.path.join(os.path.dirname(query_image), query_temp_folder_name)
    os.makedirs(query_temp_folder, exist_ok=True)
    query_basename = os.path.basename(query_image)
    destfile = os.path.join(query_temp_folder, query_basename)
    resized_image = resize_image(query_image, destfile)
    return resized_image


def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    accumulated_index_start = None
    accumulated_index_end = None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index - 1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
    return np.arange(accumulated_index_start, accumulated_index_end)


def get_locations_2_use(image_db_index, k_nearest_indices, accumulated_indexes_boundaries, query_image_locations,
                        locations_agg):
    '''
    Get a pair of locations to use, the query image to the database image with given index.
    Return: a tuple of 2 numpy arrays, the locations pair.
    '''
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(query_image_locations[i])
                locations_2_use_db.append(locations_agg[acc_index])
                break
    return np.array(locations_2_use_query), np.array(locations_2_use_db)


# locations_agg = np.concatenate([results_dict[img][0] for img in db_images])
# descriptors_agg = np.concatenate([results_dict[img][1] for img in db_images])
# accumulated_indexes_boundaries = list(accumulate([results_dict[img][0].shape[0] for img in db_images]))
# np.savetxt('locations_agg.txt', locations_agg)
# np.savetxt('descriptors_agg.txt', descriptors_agg)
# np.savetxt('accumulated_indexes_boundaries.txt', accumulated_indexes_boundaries)


def findImage(query_image):
    db_images = get_resized_db_image_paths()
    locations_agg = np.loadtxt('delf/locations_agg.txt')
    descriptors_agg = np.loadtxt('delf/descriptors_agg.txt')
    accumulated_indexes_boundaries = np.loadtxt('delf/accumulated_indexes_boundaries.txt')

    d_tree = cKDTree(descriptors_agg)


    resized_image = preprocess_query_image(query_image, query_image)

    query_image_locations, query_image_descriptors = compute_locations_and_descriptors(resized_image)

    distance_threshold = 0.8
    # K nearest neighbors
    K = 10
    distances, indices = d_tree.query(
        query_image_descriptors, distance_upper_bound=distance_threshold, k=K, n_jobs=-1)

    unique_indices = np.array(list(set(indices.flatten())))

    unique_indices.sort()
    if unique_indices[-1] == descriptors_agg.shape[0]:
        unique_indices = unique_indices[:-1]

    unique_image_indexes = np.array(
        list(set([np.argmax([np.array(accumulated_indexes_boundaries) > index])
                  for index in unique_indices])))
    unique_image_indexes

    inliers_counts = []
    img_1 = mpimg.imread(resized_image)
    for index in unique_image_indexes:
        locations_2_use_query, locations_2_use_db = get_locations_2_use(index, indices, accumulated_indexes_boundaries,query_image_locations,
                        locations_agg)
        # Perform geometric verification using RANSAC.
        _, inliers = ransac(
            (locations_2_use_db, locations_2_use_query),  # source and destination coordinates
            AffineTransform,
            min_samples=3,
            residual_threshold=20,
            max_trials=1000)

        if inliers is None or len(inliers) == 0:
            continue
        # the number of inliers as the score for retrieved images.
        inliers_counts.append({"index": index, "inliers": sum(inliers)})
        print('Найдено особых точек для изображения {} -> {}'.format(index, sum(inliers)))
        # Visualize correspondences.
        _, ax = plt.subplots()
        img_2 = mpimg.imread(db_images[index])
        inlier_idxs = np.nonzero(inliers)[0]
        plot_matches(
            ax,
            img_1,
            img_2,
            locations_2_use_db,
            locations_2_use_query,
            np.column_stack((inlier_idxs, inlier_idxs)),
            matches_color='b')
        ax.axis('off')
        ax.set_title('Установленные соотвествия')
        plt.show()

    top_match = sorted(inliers_counts, key=lambda k: k['inliers'], reverse=True)[0]

    index = top_match['index']
    locations_2_use_query, locations_2_use_db = get_locations_2_use(index, indices, accumulated_indexes_boundaries,query_image_locations,
                        locations_agg)
    # Using RANSAC.
    _, inliers = ransac(
        (locations_2_use_db, locations_2_use_query),  # source and destination coordinates
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    _, ax = plt.subplots()
    img_2 = mpimg.imread(db_images[index])
    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(
        ax,
        img_1,
        img_2,
        locations_2_use_db,
        locations_2_use_query,
        np.column_stack((inlier_idxs, inlier_idxs)),
        matches_color='b')
    ax.axis('off')
    ax.set_title('Установленные соотвествия')
    plt.show()