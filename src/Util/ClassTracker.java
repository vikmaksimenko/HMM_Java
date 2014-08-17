package Util;

public class ClassTracker {

    public int classLabel = 0;
    public int counter = 0;
    public String className = "NOT_SET";

    public ClassTracker(int classLabel, int counter, String className) {
        this.classLabel = classLabel;
        this.counter = counter;
        this.className = className;
    }

    public static boolean sortByClassLabelDescending(ClassTracker a, ClassTracker b) {
        return a.classLabel > b.classLabel;
    }

    static boolean sortByClassLabelAscending(ClassTracker a, ClassTracker b) {
        return a.classLabel < b.classLabel;
    }

    public ClassTracker(int classLabel, int i) {
        this.classLabel = classLabel;
        this.counter = i;
    }
}
