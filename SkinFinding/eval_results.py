# import needed modules from main program
from SkinFinding import *

# load the results of parameter sweep
R = pickle.load(open(".Results_Cache", "r"))
testset = glob(join("face_testing", "face*.png"))
n_test = len(testset)
si = ScreenImage()


def get_best(feature, clf, i):
    # Find the Parameter sweep index (over all features) that maximizes
    # the Jaccard score
    max_ = 0
    best = {}
    for k, r in enumerate(R):
        if feature == r["Feature"] or feature == "All":
            if clf == r["Classifier"] or clf == "All":
                if r["Score"][i] > max_:
                    max_ = r["Score"][i]
                    best["Score"] = max_
                    best["Index"] = k
    print "Image: %d, Score: %.2f, Index: %d, Feat: %s, Clf: %s" % \
        (i, best["Score"], best["Index"], feature, clf)
    return best

# Iterate over the test set and using Best_overall index
for i, testname in enumerate(testset):
    truthname = get_groundname(testname)
    im_orig = imread(testname)
    im_truth = rgb2gray(imread(truthname)).astype(np.uint8)*255
    space = ["Best RGB", "Best rg", "Best Random Forest",
             "Best Naive Bayes"]
    im_skin = [[] for k in space]
    title = ["" for k in space]

    for j, s in enumerate(space):
        if s == "Best RGB":
            best = get_best("RGB", "All", i)
        elif s == "Best rg":
            best = get_best("rg", "All", i)
        elif s == "Best Naive Bayes":
            best = get_best("All", "NB", i)
        elif s == "Best Random Forest":
            best = get_best("All", "RF", i)

        # Set Params according to Best index
        params = {}
        params["classifier"] = R[best["Index"]]["Classifier"]
        params["feature"] = R[best["Index"]]["Feature"]
        params["n_cluster"] = R[best["Index"]]["Kmeans"]
        params["thresh"] = R[best["Index"]]["Overlap_Thresh"]

        # Get feature Samples, Labels for these params
        trainset = glob(join("face_training", "face*.png"))
        Labels, Samples = get_training_samples(trainset, params)

        # Train a classifier for these Samples, Labels
        if params["classifier"] == "NB":
            clf = GaussianNB()
        elif params["classifier"] == "RF":
            clf = RandomForestClassifier()
        clf.fit(Samples, Labels)
        # Test the image for this classifier and these params
        im_lab, _, fvec = im2feature(testname, params)
        im_skin[j] = clf.predict(fvec).reshape(im_lab.shape).astype(np.uint8)

        # Report results for this test image
        score = jaccard_similarity_score(im_truth, im_skin[j], normalize=True)
        title[j] = "%s\nClassifier: %s, Space: %s\nThresh: %.2f, K: %d, Score: %.2f" \
            % (space[j], params["classifier"], params["feature"], params["thresh"],
               params["n_cluster"], score)

    si.show(testname, [im_orig, im_truth, im_skin[0], im_skin[1],
                       im_skin[2], im_skin[3]],
            ["Original\n%s" % testname, "Groundtruth", title[0], title[1],
            title[2], title[3]])
