import cv2

def brute_force_matching(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches

def flann_matching(des1, des2):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def ransac(matches, ratio=0.7):
    good_matches = []
    
    # Check if matches is a list of lists (from knnMatch)
    if isinstance(matches[0], list):
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    else:  # It's a list of DMatch objects (from brute force matching)
        good_matches = matches  # You can return all matches or apply some filtering if needed

    return good_matches
