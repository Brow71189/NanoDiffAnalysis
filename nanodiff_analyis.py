#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:13:40 2017

@author: mittelberger2
"""

import numpy as np
from JitterWizard import correct_jitter
import h5py
import os
import sys
#needed on windows
if not hasattr(sys, 'argv'):
    sys.argv = ['']
from multiprocessing import Queue, Process, set_executable, get_start_method
import queue
import threading
import time
import scipy.optimize

class NanoDiffAnalyzerWorker(object):
    def __init__(self, hdf5filequeue, outqueue, max_number_peaks, second_ring_min_distance, blur_radius,
                 noise_tolerance, length_tolerance, angle_tolerance, minimum_peak_distance, maximum_peak_radius):
        self.hdf5filequeue = hdf5filequeue
        self.outqueue = outqueue
        self.max_number_peaks = max_number_peaks
        self.second_ring_min_distance = second_ring_min_distance
        self.length_tolerance = length_tolerance
        self.angle_tolerance = angle_tolerance
        self.Jitter = correct_jitter.Jitter()
        self.blur_radius = blur_radius
        self.noise_tolerance = noise_tolerance
        self.minimum_peak_distance = minimum_peak_distance
        self.maximum_peak_radius = maximum_peak_radius
        self.image = None
        self.shape = None

    @property
    def blur_radius(self):
        return self.Jitter.blur_radius

    @blur_radius.setter
    def blur_radius(self, blur_radius):
        self.Jitter.blur_radius = blur_radius

    @property
    def noise_tolerance(self):
        return self.Jitter.noise_tolerance

    @noise_tolerance.setter
    def noise_tolerance(self, noise_tolerance):
        self.Jitter.noise_tolerance = noise_tolerance

    def _analysis_loop(self):
        while True:
            index, image = self.hdf5filequeue.get(timeout=3)
            if index is None:
                break
            res = self.analyze_nanodiff_pattern(image)
            self.outqueue.put((index,) + res)

    def analyze_nanodiff_pattern(self, image):
        self.image = image
        self.shape = self.image.shape
        self.Jitter.image = self.image
        peaks = self.Jitter.local_maxima[1]

        if len(peaks) > self.max_number_peaks:
            peaks = peaks[:self.max_number_peaks]
        first_ring, second_ring, center = self.sort_peaks(peaks)
        first_hexagon = second_hexagon = None
        if len(first_ring) > 4:
            first_hexagon = self.find_hexagon_2(first_ring, center)
            if len(first_hexagon) < 3:
                first_hexagon = None
        if len(second_ring) > 4:
            second_hexagon = self.find_hexagon_2(second_ring, center)
            if len(second_hexagon) < 3:
                second_hexagon = None

        if (second_hexagon is None and first_hexagon is not None and len(first_hexagon) > 0 and
            np.mean(np.sum((np.array(first_hexagon) - center)**2, axis=1)) > self.second_ring_min_distance*np.mean(self.shape)):
            second_hexagon = first_hexagon
            first_hexagon = None

        #remove peaks that are too close together
        to_change = []
        if first_hexagon is not None and len(first_hexagon) > 6:
            for k in range(len(first_hexagon)):
                for l in range(k+1, len(first_hexagon)):
                    distance = np.sqrt(np.sum((first_hexagon[k] - first_hexagon[l])**2))
                    if distance < self.minimum_peak_distance:
                        to_change.append((k, l))
            for entry in to_change:
                if entry[0] < len(first_hexagon) and entry[1] < len(first_hexagon):
                    first_hexagon[entry[0]] = np.rint((first_hexagon[entry[0]] + first_hexagon[entry[1]])/2).astype(np.int)
                    first_hexagon.pop(entry[1])

        to_change = []
        if second_hexagon is not None and len(second_hexagon) > 6:
            for k in range(len(second_hexagon)):
                for l in range(k+1, len(second_hexagon)):
                    distance = np.sqrt(np.sum((second_hexagon[k] - second_hexagon[l])**2))
                    if distance < self.minimum_peak_distance:
                        to_change.append((k, l))
            for entry in to_change:
                if entry[0] < len(second_hexagon) and entry[1] < len(second_hexagon):
                    second_hexagon[entry[0]] = np.rint((second_hexagon[entry[0]] + second_hexagon[entry[1]])/2).astype(np.int)
                    second_hexagon.pop(entry[1])

        #remove peaks whose intensity differs most from all others if more than 6 peaks were found for a ring
        while first_hexagon is not None and len(first_hexagon) > 6:
            largest_difference = (0, 0)
            for k in range(len(first_hexagon)):
                sum_other_peaks = 0
                for l in range(len(first_hexagon)):
                    if l != k:
                        sum_other_peaks += image[tuple(first_hexagon[l])]
                mean_other_peaks = sum_other_peaks/(len(first_hexagon)-1)
                difference = np.abs(image[tuple(first_hexagon[k])] - mean_other_peaks)
                if  difference > largest_difference[1]:
                    largest_difference = (k, difference)
            first_hexagon.pop(largest_difference[0])

        while second_hexagon is not None and len(second_hexagon) > 6:
            largest_difference = (0, 0)
            for k in range(len(second_hexagon)):
                sum_other_peaks = 0
                for l in range(len(second_hexagon)):
                    if l != k:
                        sum_other_peaks += image[tuple(second_hexagon[l])]
                mean_other_peaks = sum_other_peaks/(len(second_hexagon)-1)
                difference = np.abs(image[tuple(second_hexagon[k])] - mean_other_peaks)
                if  difference > largest_difference[1]:
                    largest_difference = (k, difference)
            second_hexagon.pop(largest_difference[0])


        return (first_hexagon, second_hexagon, center)

    def sort_peaks(self, peaks):
        peaks = np.array(peaks)
        center = peaks[0]
        peak_distance = np.sqrt(np.sum((peaks - center)**2, axis=1))
        hist = np.histogram(peak_distance, bins=8, range=(0, np.mean(self.shape)/2*self.maximum_peak_radius))
        sorted_hist = np.argsort(hist[0])
        if hist[1][sorted_hist[-1]] < hist[1][sorted_hist[-2]]:
            first_ring = (hist[1][sorted_hist[-1]], hist[1][sorted_hist[-1] + 1])
            second_ring = (hist[1][sorted_hist[-2]], hist[1][sorted_hist[-2] + 1])
        else:
            first_ring = (hist[1][sorted_hist[-2]], hist[1][sorted_hist[-2] + 1])
            second_ring = (hist[1][sorted_hist[-1]], hist[1][sorted_hist[-1] + 1])
        first_ring_peaks = []
        second_ring_peaks = []
        for i in range(len(peaks)):
            if first_ring[0] <= peak_distance[i] <= first_ring[1]:
                first_ring_peaks.append(peaks[i])
            elif second_ring[0] <= peak_distance[i] <= second_ring[1]:
                second_ring_peaks.append(peaks[i])
        first_ring_peaks_sorted = sorted(first_ring_peaks, key=lambda value: positive_angle(np.arctan2(*(value - center))))
        second_ring_peaks_sorted = sorted(second_ring_peaks, key=lambda value: positive_angle(np.arctan2(*(value - center))))
        return (first_ring_peaks_sorted, second_ring_peaks_sorted, center)
    
    def find_hexagon_2(self, peaks_sorted, center, QF_tune_factor=2):
        angle_tolerance = self.angle_tolerance/180*np.pi
        hexagon_candidates = []
        hexagon_QF = []
        for k in range(len(peaks_sorted)):
            hexagon_candidates.append([peaks_sorted[k]])
            hexagon_QF.append([12, 2, 12])
            vec0 = peaks_sorted[k] - center
            length0 = np.sqrt(np.sum(vec0**2))
            hexagon_radii = [length0]
            for i in range(len(peaks_sorted)):
                if i == k:
                    continue
                vec1 = peaks_sorted[i] - center
                length1 = np.sqrt(np.sum(vec1**2))
                if (np.abs(vec0) == np.abs(vec1)).all():
                    angle = 0
                else:
                    angle = np.arccos(np.dot(vec0, vec1)/(length0*length1))
                angle_deviation = np.pi/6 - np.abs(np.pi/6 - np.abs(angle%(np.pi/3)))
                if  angle_deviation < angle_tolerance and np.abs(length0 - length1)/length0 < self.length_tolerance:
                    hexagon_candidates[k].append(peaks_sorted[i])
                    hexagon_radii.append(length1)
                    hexagon_QF[k][0] += 12 - angle_deviation*180/np.pi
            hexagon_QF[k][0] /= len(hexagon_candidates[k])
            hexagon_QF[k][1] = QF_tune_factor*(6-np.abs(6-len(hexagon_candidates[k])))
            hexagon_QF[k][2] = 12 * (1 - (QF_tune_factor*np.std(hexagon_radii)/np.mean(hexagon_radii)))
        
        summed_hexagon_QF = np.sum(hexagon_QF, axis=1)
        best_hexagon = np.argmax(summed_hexagon_QF)
        return hexagon_candidates[best_hexagon]

    def find_hexagon(self, peaks_sorted, center):
        angle_tolerance = self.angle_tolerance/180*np.pi
        removed_peak = True
        while removed_peak:
            removed_peak = False
            hexagon = []
            peaks_added = []
            for i in range(0, len(peaks_sorted)):
                peak1 = peaks_sorted[i-2]
                peak2 = peaks_sorted[i-1]
                peak3 = peaks_sorted[i]
                edge1 = peak1 - peak2
                edge2 = peak3 - peak2
                lengths = [np.sqrt(np.sum((edge1)**2)), np.sqrt(np.sum((edge2)**2))]
                angle = np.arccos(np.dot(edge1, edge2)/(np.product(lengths)))
                radii = [np.sqrt(np.sum((peak1 - center)**2)), np.sqrt(np.sum((peak2 - center)**2)), np.sqrt(np.sum((peak3 - center)**2))]
                #print(peak1, lengths, angle*180/np.pi)
                if (np.abs(lengths[0] - lengths[1]) < self.length_tolerance*np.mean(lengths) and
                    np.abs(angle - 2*np.pi/3) < angle_tolerance):
                    peak_index = i-2 if i-2 >= 0 else len(peaks_sorted) + (i-2)
                    if np.abs(radii[0] - np.mean(radii[1:])) > self.length_tolerance*np.mean(radii):
                        peaks_sorted.pop(peak_index)
                        removed_peak = True
                        break
                    elif not peak_index in peaks_added:
                        hexagon.append(peak1)
                        peaks_added.append(peak_index)
                    peak_index = i-1 if i-1 >= 0 else len(peaks_sorted) + (i-1)
                    if np.abs(radii[1] - np.mean((radii[0], radii[2]))) > self.length_tolerance*np.mean(radii):
                        peaks_sorted.pop(peak_index)
                        removed_peak = True
                        break
                    elif not peak_index in peaks_added:
                        hexagon.append(peak2)
                        peaks_added.append(peak_index)
                    peak_index = i if i >= 0 else len(peaks_sorted) + i
                    if np.abs(radii[2] - np.mean(radii[:-1])) > self.length_tolerance*np.mean(radii):
                        peaks_sorted.pop(peak_index)
                        removed_peak = True
                        break
                    elif not peak_index in peaks_added:
                        hexagon.append(peak3)
                        peaks_added.append(peak_index)
        return hexagon

class NanoDiffAnalyzer(object):
    def __init__(self, **kwargs):
        self.filename = kwargs.get('filename')
        self.shape = kwargs.get('shape')
        self.max_number_peaks = kwargs.get('max_number_peaks', 30)
        self.second_ring_min_distance = kwargs.get('second_ring_min_distance', 0.5)
        self.maximum_peak_radius = kwargs.get('maximum_peak_radius', 1)
        self.blur_radius = kwargs.get('blur_radius', 10)
        self.noise_tolerance = kwargs.get('noise_tolerance', 1)
        self.length_tolerance = kwargs.get('length_tolerance', 0.1)
        self.angle_tolerance = kwargs.get('angle_tolerance', 5)
        self.minimum_peak_distance = kwargs.get('minimum_peak_distance', 50)
        self.first_peaks = self.second_peaks = self.centers = None
        self.number_slices = None
        self.number_processes = kwargs.get('number_processes', 3)
        self._workers = []
        self._filequeue = Queue(maxsize=20)
        self._outqueue = Queue()
        #self._manager = Manager()
        #self._stop_event = threading.Event()
        self._number_slices_set_event = threading.Event()
        self._abort_event = threading.Event()
        self.report_progress = None
        self.starttime = None

    def process_nanodiff_map(self):
        self.starttime = time.time()
        self._abort_event.clear()
        threading.Thread(target=self._fill_filequeue).start()
        self._number_slices_set_event.wait(timeout=10)
        assert self._number_slices_set_event.is_set()
        self._number_slices_set_event.clear()
        if self.shape is None:
            self.shape = (int(np.sqrt(self.number_slices)), int(np.sqrt(self.number_slices)))
        else:
            self.number_slices = np.product(self.shape)
        assert np.product(self.shape) == self.number_slices
        if self.number_processes is None or self.number_processes < 1:
            self.number_processes = os.cpu_count()
        # Needed for method "spawn" (on Windows) to prevent mutliple Swift instances from being started
        if get_start_method() == 'spawn':
            set_executable(os.path.join(sys.exec_prefix, 'python.exe'))
        for i in range(self.number_processes):
            analyzer = NanoDiffAnalyzerWorker(self._filequeue, self._outqueue,
                                              self.max_number_peaks, self.second_ring_min_distance,
                                              self.blur_radius, self.noise_tolerance, self.length_tolerance,
                                              self.angle_tolerance, self.minimum_peak_distance,
                                              self.maximum_peak_radius)
            process = Process(target=analyzer._analysis_loop)
            process.daemon = True
            time.sleep(0.1)
            process.start()
            self._workers.append(process)
            time.sleep(0.1)

        worker_handler = threading.Thread(target=self._worker_handler)
        worker_handler.daemon = True
        worker_handler.start()

        result_handler = threading.Thread(target=self._result_handler)
        result_handler.daemon = True
        result_handler.start()

        result_handler.join()
        worker_handler.join(1)
        if worker_handler.is_alive():
            self.abort()

        print(time.time() - self.starttime)

    def process_nanodiff_image(self, image):
        analyzer = NanoDiffAnalyzerWorker(self._filequeue, self._outqueue,
                                          self.max_number_peaks, self.second_ring_min_distance,
                                          self.blur_radius, self.noise_tolerance, self.length_tolerance,
                                          self.angle_tolerance, self.minimum_peak_distance,
                                          self.maximum_peak_radius)
        first_hexagon, second_hexagon, center = analyzer.analyze_nanodiff_pattern(image)
        return (first_hexagon, second_hexagon, center, analyzer.Jitter.blurred_image)

    def make_strain_map(self):
        expanded_centers = np.expand_dims(self.centers, 2)
        expanded_centers = np.repeat(expanded_centers, 6, axis=2)
        centered_peaks = self.second_peaks - expanded_centers
        radii = np.sqrt(np.sum((centered_peaks)**2, axis=-1))
        angles = np.arctan2(centered_peaks[..., 0], centered_peaks[..., 1])
        ellipse = np.zeros(self.second_peaks.shape[:2] + (3,))
        for k in range(radii.shape[0]):
            for i in range(radii.shape[1]):
                if not (self.second_peaks[k, i] == 0).all():
                    angle = angles[k,i][np.sum(self.second_peaks[k,i], axis=-1) != 0]
                    radius = radii[k,i][np.sum(self.second_peaks[k,i], axis=-1) != 0]
                    ellipse[k, i] = np.array(fit_ellipse(angle, radius))
        return ellipse

    def abort(self):
        self._abort_event.set()

    def _worker_handler(self):
        workers_finished = 0
        while workers_finished < self.number_processes:
            for worker in self._workers:
                if not worker.is_alive():
                    print('worker sleeping')
                if self._abort_event.is_set():
                    worker.terminate()
                if worker.exitcode is not None:
                    worker.join()
                    self._workers.remove(worker)
                    workers_finished += 1
                time.sleep(0.1)
            time.sleep(0.1)

    def _result_handler(self):
        self.first_peaks = np.zeros(tuple(self.shape) + (6, 2))
        self.second_peaks = np.zeros(tuple(self.shape) + (6, 2))
        self.centers = np.zeros(tuple(self.shape) + (2,))
        errorfile = open('errors.txt', 'w+')
        i = 0
        last_report_time = 0
        while i < self.number_slices and not self._abort_event.is_set():
            try:
                index, first_hexagon, second_hexagon, center = self._outqueue.get(timeout=1)
            except queue.Empty:
                pass
            else:
                if time.time() - last_report_time > 1:#i%100 == 0:
                    #print('Processed {:.0f} out of {:.0f} slices.          '.format(i, self.number_slices), end='\r')
                    if callable(self.report_progress):
                        self.report_progress(i, self.number_slices, time.time() - self.starttime)
                    last_report_time = time.time()
                x_coord = index%self.shape[1]
                y_coord = index//self.shape[1]
                try:
                    if first_hexagon is not None and len(first_hexagon) > 0:
                        self.first_peaks[y_coord, x_coord, :len(first_hexagon)] = np.array(first_hexagon)
                except Exception as e:
                    errorfile.write('{:.0f}, first hexagon: {}\n'.format(index, str(first_hexagon)))
                    print('Error in first hexagon in slice {:.0f}: {}'.format(index, str(e)))
                try:
                    if second_hexagon is not None and len(second_hexagon) > 0:
                        self.second_peaks[y_coord, x_coord, :len(second_hexagon)] = np.array(second_hexagon)
                except Exception as e:
                    errorfile.write('{:.0f}, second hexagon: {}\n'.format(index, str(second_hexagon)))
                    print('Error in second hexagon in slice {:.0f}: {}'.format(index, str(e)))
                self.centers[y_coord, x_coord] = center
                i += 1
        if callable(self.report_progress):
            self.report_progress(i, self.number_slices, time.time() - self.starttime)
        errorfile.close()

    def _fill_filequeue(self):
            _filelink = openhdf5file(self.filename)
            self.number_slices = len(_filelink['data/science_data/data'])
            self._number_slices_set_event.set()
            i = 0
            while i < self.number_slices and not self._abort_event.is_set():
                try:
                    self._filequeue.put((i, gethdf5slice(i, _filelink)), timeout=0.2)
                except queue.Full:
                    pass
                else:
                    i += 1
            _filelink.close()
            for i in range(len(self._workers)):
                self._filequeue.put((None, None))

def ellipse(polar_angle, a, b, rotation):
    """
    Returns the radius of a point lying on an ellipse with the given parameters.
    """
    return a*b/np.sqrt((b*np.cos(polar_angle-rotation))**2+(a*np.sin(polar_angle-rotation))**2)

def fit_ellipse(angles, radii):
    if len(angles) != len(radii):
        raise ValueError('The input sequences have to have the same lenght!.')
    if len(angles) < 3:
        print('Can only fit a circle and not an ellipse to a set of less than 3 points.')
        return (np.mean(radii), np.mean(radii), 0.)
    try:
        popt, pcov = scipy.optimize.curve_fit(ellipse, angles, radii, p0=(np.mean(radii), np.mean(radii), 0.0))
        axes_ratio = popt[0]/popt[1]
        if axes_ratio > 2 or axes_ratio < 0.5:
            raise RuntimeError
    except:
        print('Fit of the ellipse faied. Using a circle as best approximation of the data.')
        return (np.mean(radii), np.mean(radii), 0.)
    else:
        popt[2] %= np.pi
        return tuple(popt)

def positive_angle(angle):
    """
    Calculates the angle between 0 and 2pi from an input angle between -pi and pi (all angles in rad)
    """
    if angle < 0:
        return angle  + 2*np.pi
    else:
        return angle

def openhdf5file(hdf5filename):
    hdf5filelink=h5py.File(hdf5filename, 'r')
    return hdf5filelink

def gethdf5slice(slicenumber, hdf5filelink):
    hdf5slice = hdf5filelink['data/science_data/data'][slicenumber,:]
    return hdf5slice

if __name__ == '__main__':
    #first_peaks, second_peaks, centers = process_nanodiff_map('/3tb/nanodiffraction_maps/20161128_171207/tt.h5')
    #fp, sp, c = analyze_nanodiff_pattern(57, openhdf5file('/3tb/nanodiffraction_maps/20161128_171207/tt.h5'))
    A = NanoDiffAnalyzer(filename='/3tb/nanodiffraction_maps/20161128_152252/grating2.h5')
    res = A.process_nanodiff_map()
    np.save('/3tb/nanodiffraction_maps/peak_finding/second_peaks.npy', A.second_peaks)
    np.save('/3tb/nanodiffraction_maps/peak_finding/first_peaks.npy', A.first_peaks)
    np.save('/3tb/nanodiffraction_maps/peak_finding/centers.npy', A.centers)