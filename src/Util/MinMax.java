/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Util;

/**
 *
 * @author Пользователь
 */
public class MinMax {

    public MinMax() {
        minValue = 0;
        maxValue = 0;
    }

    public MinMax(double minValue, double maxValue) {
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    public boolean updateMinMax(double newValue) {
        if (newValue < minValue) {
            minValue = newValue;
            return true;
        }
        if (newValue > maxValue) {
            maxValue = newValue;
            return true;
        }
        return false;
    }

    public double minValue;
    public double maxValue;
}
