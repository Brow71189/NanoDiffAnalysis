NanoDiffAnalysis
============

A python module to analyze nanodiffraction maps implemented as a plugin for Nion Swift¹. It finds the 12 graphene peaks in each nanodiffraction pattern and saves them as a new data item in Nion Swift¹.
It can also create a strain map of your data which is done by fitting ellipses to the peaks found in each nanodiffraction pattern.

Usage
======

For use as a Nion Swift plugin copy the whole folder into the "plugins" folder of your installation.
In the plugin panel you can use the "Open.." button to open a new nanodiffraction map or the "select opened stack" button to continue working on a map that you have already worked on earlier.
After opening a stack you can browse through it with either the browse buttons or by entering a slice number into the respective field.
To create a virtual dark field (vdf) image draw a rectangle with the Swift processing option "Rectangle Region" on the slice image and click "Virtual DF". This procedure unfortunately blocks the main thread because of the way h5py works. So it will appear like Swift has crashed but it is actually just working on your data, so be patient!
The "Settings..." button brings up a dialog with all the options for the peak finding algorithm.
The "Pick" checkbox creates a point graphic in the vdf, the peak positions and the strain map data item. It shows the position of the currently displayed slice and can be moved to a position in these images which will update the slice image accordingly.

Installation and Requirements
=============================

Requirements
------------
* Python >= 3.5 (lower versions might work but are untested)
* numpy
* scipy
* AnalyzeMaxima (Download and Documentation at https://github.com/Brow71189/AnalyzeMaxima)

Installation
------------

If you used the "Download as ZIP" function on github to get the code make sure you rename the project folder to "JitterWizard" after extracting the ZIP-archive. If you used the "clone" function you can start right away.
Copy the project folder in your Nion Swift Plugin directory (see http://nion.com/swift/developer/introduction.html#extension-locations for the plugin directory on your OS.)
Under Linux the global plugin directory is "~/.local/share/Nion/Swift/Plugins".
Then the plugin should be loaded automatically after restarting Swift. In case it is not loading correctly, make sure to start Swift from a Terminal and check the error messages written there.

¹ www.nion.com/swift
