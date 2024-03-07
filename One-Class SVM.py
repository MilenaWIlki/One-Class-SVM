from sklearn.svm import OneClassSVM

def detect_anomalies_oneclass_svm(data):
    clf = OneClassSVM(nu=0.01)
    clf.fit(data)
    outliers = clf.predict(data)
    anomalies = data[outliers == -1]
    return anomalies

# Example usage:
data = [[1], [2], [3], [10], [15], [100]]
anomalies = detect_anomalies_oneclass_svm(data)
print("Anomalies:", anomalies)
