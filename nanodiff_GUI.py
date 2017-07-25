#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 08:36:41 2017

@author: mittelberger2
"""

import logging
import os
import time
import uuid
import numpy as np
import threading

from . import nanodiff_analyis
from . import hdf5handler
from . import vdf

from nion.ui import Dialog

class NanoDiffPanelDelegate(object):
    def __init__(self, api):
        self.__api = api
        self.panel_id = 'NanoDiff-Panel'
        self.panel_name = 'NanoDiff Analysis'
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
        self._current_slice = None
        self.filepath = None
        self.slice_image = None
        self._vdf_image = None
        self._vdf_pick_region = None
        self._results_image = None
        self._results_pick_region = None
        self._single_image_peaks = None
        self._strain_map = None
        self._strain_map_pick_region = None
        self.h5file = None
        self._last_opened_folder = ''
        self._nanodiff_analyzer = nanodiff_analyis.NanoDiffAnalyzer()
        self.settings_window_open = False

    @property
    def current_slice(self):
        return self._current_slice

    @current_slice.setter
    def current_slice(self, current_slice):
        if current_slice != self._current_slice:
            self._current_slice = current_slice
            shape = None
            if self.vdf_image is not None:
                shape = self.vdf_image.data.shape
            elif self.results_image is not None:
                shape = self.results_image.data.shape
            elif self.strain_map is not None:
                shape = self.strain_map.data.shape
            if shape is not None:
                position = ((current_slice//shape[1]+0.5)/shape[0], (current_slice%shape[1]+0.5)/shape[1])
                self.update_pick_regions(position)

    @property
    def vdf_image(self):
        if self._vdf_image is None and self.slice_image is not None:
            if self.slice_image.metadata.get('vdf_uuid'):
                self._vdf_image = self.__api.library.get_data_item_by_uuid(uuid.UUID(self.slice_image.metadata.get('vdf_uuid')))
        return self._vdf_image

    @vdf_image.setter
    def vdf_image(self, vdf_image):
        self._vdf_image = vdf_image
        if vdf_image is not None:
            update_metadata(self.slice_image, {'vdf_uuid': vdf_image.uuid.hex})

    @property
    def vdf_pick_region(self):
        if self._vdf_pick_region is None:
            if self.vdf_image is not None and self.vdf_image.metadata.get('pick_region_uuid'):
                self._vdf_pick_region = self.__api.library.get_graphic_by_uuid(uuid.UUID(self.vdf_image.metadata.get('pick_region_uuid')))
        return self._vdf_pick_region

    @vdf_pick_region.setter
    def vdf_pick_region(self, vdf_pick_region):
        self._vdf_pick_region = vdf_pick_region
        if vdf_pick_region is not None:
            update_metadata(self.vdf_image, {'pick_region_uuid': vdf_pick_region.uuid.hex})

    @property
    def results_pick_region(self):
        if self._results_pick_region is None:
            if self.results_image is not None and self.results_image.metadata.get('pick_region_uuid'):
                self._results_pick_region = self.__api.library.get_graphic_by_uuid(uuid.UUID(self.results_image.metadata.get('pick_region_uuid')))
        return self._results_pick_region

    @results_pick_region.setter
    def results_pick_region(self, results_pick_region):
        self._results_pick_region = results_pick_region
        if results_pick_region is not None:
            update_metadata(self.results_image, {'pick_region_uuid': results_pick_region.uuid.hex})

    @property
    def results_image(self):
        if self._results_image is None and self.slice_image is not None:
            if self.slice_image.metadata.get('results_uuid'):
                self._results_image = self.__api.library.get_data_item_by_uuid(uuid.UUID(self.slice_image.metadata.get('results_uuid')))
        return self._results_image

    @results_image.setter
    def results_image(self, results_image):
        self._results_image = results_image
        if results_image is not None:
            update_metadata(self.slice_image, {'results_uuid': results_image.uuid.hex})

    @property
    def single_image_peaks(self):
        if self._single_image_peaks is None and self.slice_image is not None:
            if self.slice_image.metadata.get('single_image_peaks_uuid'):
                self._single_image_peaks = self.__api.library.get_data_item_by_uuid(uuid.UUID(self.slice_image.metadata.get('single_image_peaks_uuid')))
        return self._single_image_peaks

    @single_image_peaks.setter
    def single_image_peaks(self, single_image_peaks):
        self._single_image_peaks = single_image_peaks
        if single_image_peaks is not None:
            update_metadata(self.slice_image, {'single_image_peaks_uuid': single_image_peaks.uuid.hex})

    @property
    def strain_map(self):
        if self._strain_map is None and self.slice_image is not None:
            if self.slice_image.metadata.get('strain_map_uuid'):
                self._strain_map = self.__api.library.get_data_item_by_uuid(uuid.UUID(self.slice_image.metadata.get('strain_map_uuid')))
        return self._strain_map

    @strain_map.setter
    def strain_map(self, strain_map):
        self._strain_map = strain_map
        if strain_map is not None:
            update_metadata(self.slice_image, {'strain_map_uuid': strain_map.uuid.hex})

    @property
    def strain_map_pick_region(self):
        if self._strain_map_pick_region is None:
            if self.strain_map is not None and self.strain_map.metadata.get('pick_region_uuid'):
                self._strain_map_pick_region = self.__api.library.get_graphic_by_uuid(uuid.UUID(self.strain_map.metadata.get('pick_region_uuid')))
        return self._strain_map_pick_region

    @strain_map_pick_region.setter
    def strain_map_pick_region(self, strain_map_pick_region):
        self._strain_map_pick_region = strain_map_pick_region
        if strain_map_pick_region is not None:
            update_metadata(self.strain_map, {'pick_region_uuid': strain_map_pick_region.uuid.hex})

    def create_panel_widget(self, ui, document_controller):
        self.document_controller = document_controller

        def path_finished(text):
            if len(text) > 0:
                self.filepath = text
            else:
                self.filepath = None

        def slice_number_finished(text):
            if len(text) > 0:
                try:
                    self.current_slice = int(text)
                except ValueError:
                    slice_number.text = str(self.current_slice)
                else:
                    self.update_slice_image()

        def open_button_clicked(create_new_data_item=True):
            if self.filepath is None:
                file, filter, path = document_controller._document_controller._document_window.get_file_path_dialog('Open nanodiffraction map...', self._last_opened_folder, 'HDF5 Files (*.h5);; All Files (*.*)')
                self._last_opened_folder = path
                self.filepath = file

            if not os.path.isfile(self.filepath):
                logging.warn('{} is not a file'.format(self.filepath))
                self.filepath = None
                return

            path_field.text = self.filepath

            if self.slice_image is None or (self.slice_image.metadata.get('source_file_path') != self.filepath and
                                            create_new_data_item):
                self.slice_image = self.__api.library.create_data_item()
                self.current_slice = 0
            else:
                self.current_slice = self.slice_image.metadata.get('current_slice', 0)
            self.h5file = hdf5handler.openhdf5file(self.filepath)
            self.vdf_image = None
            self.vdf_pick_region = None
            self.results_image = None
            self.results_pick_region = None
            self.single_image_peaks = None
            self.strain_map = None
            self.strain_map_pick_region = None
            self._nanodiff_analyzer.shape = None
            self.update_slice_image()
            update_metadata(self.slice_image, {'source_file_path': self.filepath})

            slice_number.text = str(self.current_slice)

        def select_button_clicked():
            data_item = document_controller.target_data_item
            if data_item.metadata.get('source_file_path'):
                self.filepath = data_item.metadata.get('source_file_path')
                self.slice_image = data_item
                if not os.path.isfile(self.filepath):
                    self.filepath = None
                open_button_clicked(create_new_data_item=False)
                if self.results_image is not None:
                    parameters = self.results_image.metadata.get('peak_finding_parameters')
                    if parameters is not None:
                        for key, value in parameters.items():
                            setattr(self._nanodiff_analyzer, key, value)
                    if self.settings_window_open:
                        self.settings_window.update_fields()
                elif self.single_image_peaks is not None:
                    parameters = self.single_image_peaks.metadata.get('peak_finding_paramters')
                    if parameters is not None:
                        for key, value in parameters.items():
                            setattr(self._nanodiff_analyzer, key, value)
                    if self.settings_window_open:
                        self.settings_window.update_fields()

        def last_button_clicked():
            self.current_slice -= 1
            self.update_slice_image()
            slice_number.text = str(self.current_slice)

        def next_button_clicked():
            self.current_slice += 1
            self.update_slice_image()
            slice_number.text = str(self.current_slice)

        def last10_button_clicked():
            self.current_slice -= 10
            self.update_slice_image()
            slice_number.text = str(self.current_slice)

        def next10_button_clicked():
            self.current_slice += 10
            self.update_slice_image()
            slice_number.text = str(self.current_slice)

        def vdf_pick_region_changed(key):
            if key == 'position':
                position = self.vdf_pick_region.position
                self.update_pick_regions(position)

        def vdf_pick_region_deleted():
            pick_checkbox.checked = False

        def results_pick_region_changed(key):
            if key == 'position':
                position = self.results_pick_region.position
                self.update_pick_regions(position)

        def results_pick_region_deleted():
            pick_checkbox.checked = False
        
        def strain_map_pick_region_changed(key):
            if key == 'position':
                position = self.strain_map_pick_region.position
                self.update_pick_regions(position)

        def strain_map_pick_region_deleted():
            pick_checkbox.checked = False

        def pick_checkbox_changed(check_state):
            if check_state == 'checked':
                if self.vdf_image is not None:
                    if self.vdf_pick_region is None:
                        x_coord = self.current_slice%self.vdf_image.data.shape[1]
                        y_coord = self.current_slice//self.vdf_image.data.shape[1]
                        self.vdf_pick_region = self.vdf_image.add_point_region((y_coord+0.5)/self.vdf_image.data.shape[0], (x_coord+0.5)/self.vdf_image.data.shape[1])
                        self.vdf_pick_region.set_property('is_bounds_constrained', True)
                        self.vdf_pick_region.label = 'Pick'
                    property_changed_event = self.vdf_pick_region.get_property('property_changed_event')
                    region_deleted_event = self.vdf_pick_region.get_property('about_to_be_removed_event')
                    self.vdf_changed_event_listener = property_changed_event.listen(vdf_pick_region_changed)
                    self.vdf_deleted_event_listener = region_deleted_event.listen(vdf_pick_region_deleted)
                if self.results_image is not None:
                    if self.results_pick_region is None:
                        x_coord = self.current_slice%self.results_image.data.shape[-1]
                        y_coord = self.current_slice//self.results_image.data.shape[-1]
                        self.results_pick_region = self.results_image.add_point_region((y_coord+0.5)/self.results_image.data.shape[-2], (x_coord+0.5)/self.results_image.data.shape[-1])
                        self.results_pick_region.set_property('is_bounds_constrained', True)
                        self.results_pick_region.label = 'Pick'
                    property_changed_event = self.results_pick_region.get_property('property_changed_event')
                    region_deleted_event = self.results_pick_region.get_property('about_to_be_removed_event')
                    self.results_changed_event_listener = property_changed_event.listen(results_pick_region_changed)
                    self.results_deleted_event_listener = region_deleted_event.listen(results_pick_region_deleted)
                if self.strain_map is not None:
                    if self.strain_map_pick_region is None:
                        x_coord = self.current_slice%self.strain_map.data.shape[1]
                        y_coord = self.current_slice//self.strain_map.data.shape[1]
                        self.strain_map_pick_region = self.strain_map.add_point_region((y_coord+0.5)/self.strain_map.data.shape[0], (x_coord+0.5)/self.strain_map.data.shape[0])
                        self.strain_map_pick_region.set_property('is_bounds_constrained', True)
                        self.strain_map_pick_region.label = 'Pick'
                    property_changed_event = self.strain_map_pick_region.get_property('property_changed_event')
                    region_deleted_event = self.strain_map_pick_region.get_property('about_to_be_removed_event')
                    self.strain_map_changed_event_listener = property_changed_event.listen(strain_map_pick_region_changed)
                    self.strain_map_deleted_event_listener = region_deleted_event.listen(strain_map_pick_region_deleted)
            else:
                if self.vdf_image is not None and self.vdf_pick_region is not None:
                    try:
                        self.vdf_image.remove_region(self.vdf_pick_region)
                    except Exception as e:
                        print(e)
                    self.vdf_pick_region = None
                    remove_from_metadata(self.vdf_image, 'pick_region_uuid')
                    delattr(self, 'vdf_changed_event_listener')
                    delattr(self, 'vdf_deleted_event_listener')

                if self.results_image is not None and self.results_pick_region is not None:
                    try:
                        self.results_image.remove_region(self.results_pick_region)
                    except Exception as e:
                        print(e)
                    self.results_pick_region = None
                    remove_from_metadata(self.results_image, 'pick_region_uuid')
                    delattr(self, 'results_changed_event_listener')
                    delattr(self, 'results_deleted_event_listener')
                    
                if self.strain_map is not None and self.strain_map_pick_region is not None:
                    try:
                        self.strain_map.remove_region(self.strain_map_pick_region)
                    except Exception as e:
                        print(e)
                    self.strain_map_pick_region = None
                    remove_from_metadata(self.strain_map, 'pick_region_uuid')
                    delattr(self, 'strain_map_changed_event_listener')
                    delattr(self, 'strain_map_deleted_event_listener')

        def start_button_clicked():
            roi = {}
            for region in self.slice_image.regions:
                if region.type == 'rectangle-region':
                    roi['center'] = region.get_property('center')
                    roi['size'] = region.get_property('size')
                    roi['type'] = region.type
                    break
            if not roi.get('center'):
                logging.warn('You have to provide a rectangle-region to do vdf.')
                return

            def run_vdf():
                starttime = time.time()
                result = vdf.vdf(self.h5file, vdf.getroirange(self.h5file, roi))
                def write_log():
                    start_button._PushButtonWidget__push_button_widget.enabled = True
                    logging.info('Processing time (hdf5): %.2f s.' % (time.time()-starttime,))
                self.__api.queue_task(write_log)
                if self.vdf_image is None:
                    def create_item():
                        self.vdf_image = self.__api.library.create_data_item()
                    self.__api.queue_task(create_item)
                def update_item():
                    self.update_vdf_image(result, roi)
                self.__api.queue_task(update_item)
                self.__api.queue_task(lambda: update_metadata(self.vdf_image, {'source_uuid': self.slice_image.uuid.hex}))
            self.vdf_thread = threading.Thread(target=run_vdf)
            self.vdf_thread.start()
            #start_button._PushButtonWidget__push_button_widget.enabled = False

        def find_peaks_button_clicked():
            if find_peaks_button.text == 'Abort':
                self._nanodiff_analyzer.abort()
                return

            if self.slice_image is None:
                logging.warn('You have to open or select a hdf5 stack first.')
                return

            def run_find_peaks():
                self._nanodiff_analyzer.filename = self.filepath
                try:
                    self._nanodiff_analyzer.process_nanodiff_map()
                except AssertionError:
                    self._nanodiff_analyzer.shape = None
                    self.__api.queue_task(logging.warn('The number of slices does not match the shape of the map. ' + 
                                                    'Maybe it is a non-square map? Try setting its shape explicitly.'))
                    return
                finally:
                    def update_text():
                        find_peaks_button.text = 'Find peaks stack'
                    self.__api.queue_task(update_text)
                if self.results_image is None:
                    def create_item():
                        self.results_image = self.__api.library.create_data_item()
                    self.__api.queue_task(create_item)
                def update_item():
                    self.update_results_image()
                self.__api.queue_task(update_item)
                self.__api.queue_task(lambda: update_metadata(self.results_image, {'source_uuid': self.slice_image.uuid.hex}))

            self.find_peaks_thread = threading.Thread(target=run_find_peaks)
            self.find_peaks_thread.start()
            find_peaks_button.text = 'Abort'

        def find_peaks_single_button_clicked():
            first_hexagon, second_hexagon, center, blurred_image = self._nanodiff_analyzer.process_nanodiff_image(self.slice_image.data)
            if self.single_image_peaks is None:
                self.single_image_peaks = self.__api.library.create_data_item()
            self.update_single_image_peaks(first_hexagon, second_hexagon, center, blurred_image)
            update_metadata(self.single_image_peaks, {'source_uuid': self.slice_image.uuid.hex})

        def make_strain_map_button_clicked():
            if self.results_image is not None:
                if self._nanodiff_analyzer.second_peaks is None:
                    second_peaks = self.results_image.data[6:12]
                    second_peaks = np.moveaxis(second_peaks, 0, -1)
                    second_peaks = np.moveaxis(second_peaks, 0, -1)
                    self._nanodiff_analyzer.second_peaks = second_peaks
                    centers = self.results_image.data[12]
                    centers = np.moveaxis(centers, 0, -1)
                    self._nanodiff_analyzer.centers = centers
                if self.strain_map is None:
                    self.strain_map = self.__api.library.create_data_item()
                data = self._nanodiff_analyzer.make_strain_map()
                self.update_strain_map(data)
                update_metadata(self.strain_map, {'source_uuid': self.slice_image.uuid.hex})
            
        def pretty_time_format(time_s):
            time_s = int(time_s)
            h = time_s//3600
            r_h = time_s%3600
            m = r_h//60
            s = r_h%60
            
            pretty_string = ''
            if h != 0:
                pretty_string += '{:d}h '.format(h)
            
            if m != 0 or h != 0:
                pretty_string += '{:02d}m '.format(m)
            
            pretty_string += '{:02d}s'.format(s)
            
            return pretty_string
            
            
        def update_progress_label(slices_done, total_number_slices, runtime):
            
            if slices_done > 0:
                eta = runtime/slices_done*total_number_slices - runtime
                eta_string = pretty_time_format(eta)
            else:
                eta_string = '--'
                
            time_string = pretty_time_format(runtime)
            space = len(str(total_number_slices))
            def update_labels():
                progress_label.text = ('{:>'+ str(space) +'.0f}/{:.0f}').format(slices_done, total_number_slices)
                
                time_label.text = '{:s}  ETA: {:s}'.format(time_string, eta_string)
            
            document_controller.queue_task(update_labels)
        
        self._nanodiff_analyzer.report_progress = update_progress_label

        column = ui.create_column_widget()
        descriptor_row1 = ui.create_row_widget()
        descriptor_row1.add(ui.create_label_widget("Path to HDF5-file:"))

        parameters_row1 = ui.create_row_widget()
        path_field = ui.create_line_edit_widget()
        path_field.on_editing_finished = path_finished
        parameters_row1.add(path_field)
        parameters_row1.add_spacing(15)
        open_button = ui.create_push_button_widget("Open...")
        open_button.on_clicked = open_button_clicked
        parameters_row1.add(open_button)
        parameters_row1.add_spacing(5)

        button_row0 = ui.create_row_widget()
        select_button = ui.create_push_button_widget('Select opened stack')
        select_button.on_clicked = select_button_clicked
        button_row0.add_stretch()
        button_row0.add(select_button)
        button_row0.add_spacing(5)

        descriptor_row3 = ui.create_row_widget()
        descriptor_row3.add(ui.create_label_widget("Browse through hdf5-file: "))

        button_row1 = ui.create_row_widget()
        last10_button = ui.create_push_button_widget("<<")
        last10_button.on_clicked = last10_button_clicked
        button_row1.add(last10_button)
        button_row1.add_spacing(2)

        last_button = ui.create_push_button_widget("<")
        last_button.on_clicked = last_button_clicked
        button_row1.add(last_button)
        button_row1.add_spacing(8)

        next_button = ui.create_push_button_widget(">")
        next_button.on_clicked = next_button_clicked
        button_row1.add(next_button)
        button_row1.add_spacing(2)

        next10_button = ui.create_push_button_widget(">>")
        next10_button.on_clicked = next10_button_clicked
        button_row1.add(next10_button)
        button_row1.add_spacing(5)

        parameters_row3 = ui.create_row_widget()
        parameters_row3.add(ui.create_label_widget("Jump to slice #: "))
        slice_number = ui.create_line_edit_widget()
        slice_number.on_editing_finished = slice_number_finished
        self.slice_number = slice_number
        parameters_row3.add(slice_number)
        parameters_row3.add(ui.create_label_widget(" current slice #"))
        parameters_row3.add_stretch()
        parameters_row3.add_spacing(5)

        checkbox_row = ui.create_row_widget()
        checkbox_row.add(ui.create_label_widget('Pick '))
        pick_checkbox = ui.create_check_box_widget()
        pick_checkbox.on_check_state_changed = pick_checkbox_changed
        config_button = ui.create_push_button_widget('Settings...')
        config_button.on_clicked = self.show_config_box
        checkbox_row.add(pick_checkbox)
        checkbox_row.add_stretch()
        checkbox_row.add(config_button)
        checkbox_row.add_spacing(5)
        
        progress_row = ui.create_row_widget()
        progress_row.add(ui.create_label_widget('Progress: '))
        progress_label = ui.create_label_widget()
        progress_row.add(progress_label)
        progress_row.add_stretch()
        progress_row.add_spacing(5)
        progress_row.add(ui.create_label_widget('Time: '))
        time_label = ui.create_label_widget()
        progress_row.add(time_label)
        progress_row.add_spacing(5)

        button_row2 = ui.create_row_widget()
        start_button = ui.create_push_button_widget("Virtual DF")
        start_button.on_clicked = start_button_clicked
        find_peaks_single_button = ui.create_push_button_widget("Find peaks single")
        find_peaks_single_button.on_clicked = find_peaks_single_button_clicked
        find_peaks_button = ui.create_push_button_widget("Find peaks stack")
        find_peaks_button.on_clicked = find_peaks_button_clicked
        button_row2.add(start_button)
        button_row2.add_spacing(3)
        button_row2.add(find_peaks_single_button)
        button_row2.add_spacing(3)
        button_row2.add(find_peaks_button)
        button_row2.add_spacing(5)

        button_row3 = ui.create_row_widget()
        make_strain_map_button = ui.create_push_button_widget('Make strain map')
        make_strain_map_button.on_clicked = make_strain_map_button_clicked
        button_row3.add(make_strain_map_button)
        button_row3.add_stretch()

        column.add_spacing(10)
        column.add(descriptor_row1)
        column.add_spacing(3)
        column.add(parameters_row1)
        column.add_spacing(5)
        column.add(button_row0)
        column.add_spacing(8)
        column.add(descriptor_row3)
        column.add_spacing(3)
        column.add(button_row1)
        column.add_spacing(8)
        column.add(parameters_row3)
        column.add_spacing(8)
        column.add(checkbox_row)
        column.add_spacing(15)
        column.add(progress_row)
        column.add_spacing(15)
        column.add(button_row2)
        column.add_spacing(15)
        column.add(button_row3)
        column.add_stretch()

        return column

    def update_slice_image(self):
        if self.current_slice != self.slice_image.metadata.get('current_slice'):
            self.slice_image.set_data(hdf5handler.gethdf5slice(self.current_slice, self.h5file))
            self.slice_image.title = 'Slice_{:.0f}_of_{}'.format(self.current_slice, os.path.splitext(os.path.split(self.filepath)[1])[0])
            update_metadata(self.slice_image, {'current_slice': self.current_slice})

    def update_vdf_image(self, data, roi):
        self.vdf_image.set_data(data)
        self.vdf_image.title = 'VDF_of_{}_({:.2f}_{:.2f})'.format(os.path.splitext(os.path.split(self.filepath)[1])[0], *roi['center'])

    def update_results_image(self):
        data = np.append(self._nanodiff_analyzer.first_peaks, self._nanodiff_analyzer.second_peaks, axis=-2)
        centers = self._nanodiff_analyzer.centers[..., np.newaxis, :]
        data = np.append(data, centers, axis=-2)
        data = np.moveaxis(data, 0, -1)
        data = np.moveaxis(data, 0, -1)
        data_descriptor = self.__api.create_data_descriptor(is_sequence=False, collection_dimension_count=2, datum_dimension_count=2)
        xdata = self.__api.create_data_and_metadata(data, data_descriptor=data_descriptor)
        self.results_image.set_data_and_metadata(xdata)
        self.results_image.title = 'Peak_positions_of_{}'.format(os.path.splitext(os.path.split(self.filepath)[1])[0])
        self.results_image._data_item.caption = AXES_DESCRIPTION
        parameters = {'max_number_peaks': self._nanodiff_analyzer.max_number_peaks,
                      'second_ring_min_distance': self._nanodiff_analyzer.second_ring_min_distance,
                      'blur_radius': self._nanodiff_analyzer.blur_radius,
                      'noise_tolerance': self._nanodiff_analyzer.noise_tolerance,
                      'length_tolerance': self._nanodiff_analyzer.length_tolerance,
                      'angle_tolerance': self._nanodiff_analyzer.angle_tolerance,
                      'minimum_peak_distance': self._nanodiff_analyzer.minimum_peak_distance,
                      'maximum_peak_radius': self._nanodiff_analyzer.maximum_peak_radius}
        update_metadata(self.results_image, {'peak_finding_parameters': parameters})

    def update_single_image_peaks(self, first_hexagon, second_hexagon, center, blurred_image):
        self.single_image_peaks.set_data(blurred_image)
        self.single_image_peaks.title = 'Peak_positions_of_{}'.format(self.slice_image.title)
        parameters = {'max_number_peaks': self._nanodiff_analyzer.max_number_peaks,
                      'second_ring_min_distance': self._nanodiff_analyzer.second_ring_min_distance,
                      'blur_radius': self._nanodiff_analyzer.blur_radius,
                      'noise_tolerance': self._nanodiff_analyzer.noise_tolerance,
                      'length_tolerance': self._nanodiff_analyzer.length_tolerance,
                      'angle_tolerance': self._nanodiff_analyzer.angle_tolerance,
                      'minimum_peak_distance': self._nanodiff_analyzer.minimum_peak_distance,
                      'maximum_peak_radius': self._nanodiff_analyzer.maximum_peak_radius}
        update_metadata(self.single_image_peaks, {'peak_finding_parameters': parameters})

        for region in self.single_image_peaks.regions:
            if region.type == 'point-region':
                self.single_image_peaks.remove_region(region)
        shape = self.single_image_peaks.data.shape
        if center is not None and not (np.array(center) == 0).all():
            region = self.single_image_peaks.add_point_region(center[0]/shape[0], center[1]/shape[1])
            region.label = 'center'
        if first_hexagon is not None:
            for i in range(len(first_hexagon)):
                peak = first_hexagon[i]
                if not (peak == 0).all():
                    region = self.single_image_peaks.add_point_region(peak[0]/shape[0], peak[1]/shape[1])
                    region.label = str(i+1)
        if second_hexagon is not None:
            for i in range(len(second_hexagon)):
                peak = second_hexagon[i]
                if not (peak == 0).all():
                    region = self.single_image_peaks.add_point_region(peak[0]/shape[0], peak[1]/shape[1])
                    region.label = str(i+7)

    def update_strain_map(self, data):
        #data = np.moveaxis(data, -1, 0)
        data_descriptor = self.__api.create_data_descriptor(is_sequence=False, collection_dimension_count=2, datum_dimension_count=1)
        xdata = self.__api.create_data_and_metadata(data, data_descriptor=data_descriptor)
        self.strain_map.set_data_and_metadata(xdata)
        self.strain_map.title = 'Strain_map_of_{}'.format(os.path.splitext(os.path.split(self.filepath)[1])[0])
        self.strain_map._data_item.caption = STRAIN_MAP_AXES_DESCRIPTION
        parameters = {'max_number_peaks': self._nanodiff_analyzer.max_number_peaks,
                      'second_ring_min_distance': self._nanodiff_analyzer.second_ring_min_distance,
                      'blur_radius': self._nanodiff_analyzer.blur_radius,
                      'noise_tolerance': self._nanodiff_analyzer.noise_tolerance,
                      'length_tolerance': self._nanodiff_analyzer.length_tolerance,
                      'angle_tolerance': self._nanodiff_analyzer.angle_tolerance,
                      'minimum_peak_distance': self._nanodiff_analyzer.minimum_peak_distance,
                      'maximum_peak_radius': self._nanodiff_analyzer.maximum_peak_radius}
        update_metadata(self.strain_map, {'peak_finding_parameters': parameters})
    
    def update_pick_regions(self, position):
        current_slice = None
        if self.results_pick_region is not None and not np.isclose(self.results_pick_region.position, position).all():
            current_slice = current_slice or int(position[0]*self.results_image.data.shape[-2])*self.results_image.data.shape[-1] + int(position[1]*self.results_image.data.shape[-1])
            self.results_pick_region.position = position
        if self.vdf_pick_region is not None and not np.isclose(self.vdf_pick_region.position, position).all():
            current_slice = current_slice or int(position[0]*self.vdf_image.data.shape[0])*self.vdf_image.data.shape[1] + int(position[1]*self.vdf_image.data.shape[1])
            self.vdf_pick_region.position = position
        if self.strain_map_pick_region is not None and not np.isclose(self.strain_map_pick_region.position, position).all():
            current_slice = current_slice or int(position[0]*self.strain_map.data.shape[0])*self.strain_map.data.shape[1] + int(position[1]*self.strain_map.data.shape[1])
            self.strain_map_pick_region.position = position
            
        if current_slice is not None:
            self._current_slice = current_slice
            self.slice_number.text = str(self._current_slice)
            self.update_slice_image()

    def show_config_box(self):
        dc = self.document_controller._document_controller

        class ConfigDialog(Dialog.OkCancelDialog):

            def __init__(self, ui, nanodiffGUI):
                super(ConfigDialog, self).__init__(ui, include_cancel=False)
                def report_window_close():
                    nanodiffGUI.settings_window_open = False
                    if hasattr(nanodiffGUI, 'settings_window'):
                        delattr(nanodiffGUI, 'settings_window')
                self.on_accept = report_window_close
                self.on_reject = report_window_close
                self.shape = [None, None]

                def blur_radius_finished(text):
                    if len(text) > 0:
                        try:
                            blur_radius = float(text)
                        except ValueError:
                            blur_radius_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.blur_radius)
                        else:
                            nanodiffGUI._nanodiff_analyzer.blur_radius = blur_radius
                    else:
                        blur_radius_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.blur_radius)

                def noise_tolerance_finished(text):
                    if len(text) > 0:
                        try:
                            noise_tolerance = float(text)
                        except ValueError:
                            noise_tolerance_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.noise_tolerance)
                        else:
                            nanodiffGUI._nanodiff_analyzer.noise_tolerance = noise_tolerance
                    else:
                        noise_tolerance_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.noise_tolerance)

                def max_number_peaks_finished(text):
                    if len(text) > 0:
                        try:
                            max_number_peaks = int(text)
                        except ValueError:
                            max_number_peaks_field.text = '{:.0f}'.format(nanodiffGUI._nanodiff_analyzer.max_number_peaks)
                        else:
                            nanodiffGUI._nanodiff_analyzer.max_number_peaks = max_number_peaks
                    else:
                        max_number_peaks_field.text = '{:.0f}'.format(nanodiffGUI._nanodiff_analyzer.max_number_peaks)

                def second_ring_min_distance_finished(text):
                    if len(text) > 0:
                        try:
                            second_ring_min_distance = float(text)
                        except ValueError:
                            second_ring_min_distance_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.second_ring_min_distance)
                        else:
                            nanodiffGUI._nanodiff_analyzer.second_ring_min_distance = second_ring_min_distance
                    else:
                        second_ring_min_distance_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.second_ring_min_distance)

                def length_tolerance_finished(text):
                    if len(text) > 0:
                        try:
                            length_tolerance = float(text)
                        except ValueError:
                            length_tolerance_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.length_tolerance)
                        else:
                            nanodiffGUI._nanodiff_analyzer.length_tolerance = length_tolerance
                    else:
                        length_tolerance_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.length_tolerance)

                def angle_tolerance_finished(text):
                    if len(text) > 0:
                        try:
                            angle_tolerance = float(text)
                        except ValueError:
                            angle_tolerance_field.text = '{:.1f}'.format(nanodiffGUI._nanodiff_analyzer.angle_tolerance)
                        else:
                            nanodiffGUI._nanodiff_analyzer.angle_tolerance = angle_tolerance
                    else:
                        angle_tolerance_field.text = '{:.1f}'.format(nanodiffGUI._nanodiff_analyzer.angle_tolerance)

                def minimum_peak_distance_finished(text):
                    if len(text) > 0:
                        try:
                            minimum_peak_distance = int(text)
                        except ValueError:
                            minimum_peak_distance_field.text = '{:.0f}'.format(nanodiffGUI._nanodiff_analyzer.minimum_peak_distance)
                        else:
                            nanodiffGUI._nanodiff_analyzer.minimum_peak_distance = minimum_peak_distance
                    else:
                        minimum_peak_distance_field.text = '{:.0f}'.format(nanodiffGUI._nanodiff_analyzer.minimum_peak_distance)
                        
                def maximum_peak_radius_finished(text):
                    if len(text) > 0:
                        try:
                            maximum_peak_radius = float(text)
                        except ValueError:
                            maximum_peak_radius_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.maximum_peak_radius)
                        else:
                            nanodiffGUI._nanodiff_analyzer.maximum_peak_radius = maximum_peak_radius
                    else:
                        maximum_peak_radius_field.text = '{:.2f}'.format(nanodiffGUI._nanodiff_analyzer.maximum_peak_radius)

                def number_processes_finished(text):
                    if len(text) > 0:
                        try:
                            number_processes = int(text)
                        except ValueError:
                            number_processes_field.text = '{:.0f}'.format(nanodiffGUI._nanodiff_analyzer.number_processes)
                        else:
                            nanodiffGUI._nanodiff_analyzer.number_processes = number_processes
                    else:
                        number_processes_field.text = '{:.0f}'.format(nanodiffGUI._nanodiff_analyzer.number_processes)
                
                def shape_y_finished(text):
                    if len(text) > 0:
                        try:
                            shape_y = int(text)
                        except ValueError:
                            shape_y_field.placeholder_text = 'None'
                            self.shape[0] = None
                        else:
                            self.shape[0] = shape_y
                    else:
                        shape_y_field.placeholder_text = 'None'
                        self.shape[0] = None
                    if not None in self.shape:
                        nanodiffGUI._nanodiff_analyzer.shape = tuple(self.shape)
                    else:
                        nanodiffGUI._nanodiff_analyzer.shape = None
                
                def shape_x_finished(text):
                    if len(text) > 0:
                        try:
                            shape_x = int(text)
                        except ValueError:
                            shape_x_field.placeholder_text = 'None'
                            self.shape[1] = None
                        else:
                            self.shape[1] = shape_x
                    else:
                        shape_x_field.placeholder_text = 'None'
                        self.shape[1] = None
                    if not None in self.shape:
                        nanodiffGUI._nanodiff_analyzer.shape = tuple(self.shape)
                    else:
                        nanodiffGUI._nanodiff_analyzer.shape = None

                row1 = self.ui.create_row_widget()
                row2 = self.ui.create_row_widget()
                row3 = self.ui.create_row_widget()
                row4 = self.ui.create_row_widget()
                row5 = self.ui.create_row_widget()
                row6 = self.ui.create_row_widget()
                row7 = self.ui.create_row_widget()
                row8 = self.ui.create_row_widget()
                row9 = self.ui.create_row_widget()
                row10 = self.ui.create_row_widget()
                row11 = self.ui.create_row_widget()
                row12 = self.ui.create_row_widget()

                blur_radius_field = self.ui.create_line_edit_widget()
                blur_radius_field.on_editing_finished = blur_radius_finished

                noise_tolerance_field = self.ui.create_line_edit_widget()
                noise_tolerance_field.on_editing_finished = noise_tolerance_finished

                max_number_peaks_field = self.ui.create_line_edit_widget()
                max_number_peaks_field.on_editing_finished = max_number_peaks_finished

                second_ring_min_distance_field = self.ui.create_line_edit_widget()
                second_ring_min_distance_field.on_editing_finished = second_ring_min_distance_finished

                length_tolerance_field = self.ui.create_line_edit_widget()
                length_tolerance_field.on_editing_finished = length_tolerance_finished

                angle_tolerance_field = self.ui.create_line_edit_widget()
                angle_tolerance_field.on_editing_finished = angle_tolerance_finished

                minimum_peak_distance_field = self.ui.create_line_edit_widget()
                minimum_peak_distance_field.on_editing_finished = minimum_peak_distance_finished
                
                maximum_peak_radius_field = self.ui.create_line_edit_widget()
                maximum_peak_radius_field.on_editing_finished = maximum_peak_radius_finished

                number_processes_field = self.ui.create_line_edit_widget()
                number_processes_field.on_editing_finished = number_processes_finished
                
                shape_y_field = self.ui.create_line_edit_widget()
                shape_y_field.on_editing_finished = shape_y_finished
                shape_x_field = self.ui.create_line_edit_widget()
                shape_x_field.on_editing_finished = shape_x_finished

                row1.add_spacing(5)
                row1.add(self.ui.create_label_widget('Parameters for inital peak finding:'))
                row1.add_spacing(5)
                row1.add_stretch()

                row2.add_spacing(5)
                row2.add(self.ui.create_label_widget('Blur radius (px): '))
                row2.add(blur_radius_field)
                row2.add_spacing(5)
                row2.add(self.ui.create_label_widget('Noise tolerance: '))
                row2.add(noise_tolerance_field)
                row2.add_spacing(5)
                row2.add_stretch()

                row3.add_spacing(5)
                row3.add(self.ui.create_label_widget('Parameters for finding hexagons in initial points:'))
                row3.add_spacing(5)
                row3.add_stretch()

                row4.add_spacing(5)
                row4.add(self.ui.create_label_widget('Maximum number of peaks to consider for finding hexagons: '))
                row4.add(max_number_peaks_field)
                row4.add_spacing(5)
                row4.add_stretch()

                row5.add_spacing(5)
                row5.add(self.ui.create_label_widget('Minimum distance of second ring from center (relative to image radius): '))
                row5.add(second_ring_min_distance_field)
                row5.add_spacing(5)
                row5.add_stretch()

                row6.add_spacing(5)
                row6.add(self.ui.create_label_widget('Length tolerance for comparing distance to center for peaks within one hexagon (relative): '))
                row6.add(length_tolerance_field)
                row6.add_spacing(5)
                row6.add_stretch()

                row7.add_spacing(5)
                row7.add(self.ui.create_label_widget('Angle tolerance between peaks within one hexagon (deg): '))
                row7.add(angle_tolerance_field)
                row7.add_spacing(5)
                row7.add_stretch()

                row8.add_spacing(5)
                row8.add(self.ui.create_label_widget('Peaks separated by less than this number of pixels will be considered as one peak: '))
                row8.add(minimum_peak_distance_field)
                row8.add_spacing(5)
                row8.add_stretch()
                
                row9.add_spacing(5)
                row9.add(self.ui.create_label_widget('Maximum radius for peaks to be included (relative to image radius): '))
                row9.add(maximum_peak_radius_field)
                row9.add_spacing(5)
                row9.add_stretch()

                row10.add_spacing(5)
                row10.add(self.ui.create_label_widget('Additional parameters:'))
                row10.add_spacing(5)
                row10.add_stretch()

                row11.add_spacing(5)
                row11.add(self.ui.create_label_widget('Number of processor cores to use for map analysis: '))
                row11.add(number_processes_field)
                row11.add_spacing(5)
                row11.add_stretch()
                
                row12.add_spacing(5)
                row12.add(self.ui.create_label_widget('Shape of the map (only needed for non-square maps) (y, x): '))
                row12.add(shape_y_field)
                row12.add_spacing(5)
                row12.add(shape_x_field)
                row12.add_spacing(5)
                row12.add_stretch()

                self.content.add_spacing(5)
                self.content.add(row1)
                self.content.add_spacing(15)
                self.content.add(row2)
                self.content.add_spacing(30)
                self.content.add(row3)
                self.content.add_spacing(15)
                self.content.add(row4)
                self.content.add_spacing(5)
                self.content.add(row5)
                self.content.add_spacing(5)
                self.content.add(row6)
                self.content.add_spacing(5)
                self.content.add(row7)
                self.content.add_spacing(5)
                self.content.add(row8)
                self.content.add_spacing(5)
                self.content.add(row9)
                self.content.add_spacing(30)
                self.content.add(row10)
                self.content.add_spacing(15)
                self.content.add(row11)
                self.content.add_spacing(5)
                self.content.add(row12)
                self.content.add_spacing(5)                
                self.content.add_stretch()

                def update_fields():
                    blur_radius_finished('')
                    noise_tolerance_finished('')
                    max_number_peaks_finished('')
                    second_ring_min_distance_finished('')
                    length_tolerance_finished('')
                    angle_tolerance_finished('')
                    number_processes_finished('')
                    minimum_peak_distance_finished('')
                    maximum_peak_radius_finished('')
                    shape_y_finished('')
                    shape_x_finished('')
                self.update_fields = update_fields
                update_fields()

        if not self.settings_window_open:
            self.settings_window_open = True
            self.settings_window = ConfigDialog(dc.ui, self)
            self.settings_window.show()

def update_metadata(data_item, new_metadata):
    metadata = data_item.metadata
    metadata.update(new_metadata)
    data_item.set_metadata(metadata)

def remove_from_metadata(data_item, key):
    metadata = data_item.metadata
    metadata.pop(key, None)
    data_item.set_metadata(metadata)

AXES_DESCRIPTION = """Axes description:
    1. Peak index (0-5: inner ring, 6-12: outer ring)
    2. Peak y-, and x-coordinate
    3. Map y-coordinate
    4. Map x-coordinate"""

STRAIN_MAP_AXES_DESCRIPTION = """Axes description:
    1. Ellipse parameters (axis 1, axis 2, angle)
    2. Map y-coordinate
    3. Map x-coordinate"""

class NanoDiffExtension(object):
    extension_id = 'univie.nanodiff'

    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(NanoDiffPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None
