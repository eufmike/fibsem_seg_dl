# DL Segmentation on FIB-SEM Datasets
This repository hosts code for performing deep-learning based segmentation in the Washington University Center for Cellular Imaging. <br/>

This new repository is prepared for WU RIS HPC. <br/>

## Information
* By Michael Chien-Cheng Shih, PhD, Staff Scientist
* email: m.cc.shih@gmail.com
* Date: July 14, 2020
* GitHub Repository: https://github.com/WUCCI-WashU

## Datasets
* Data: Generated from Zeiss Crossbeam 540 FIB-SEM
    * 3D dataset
    * Voxel-size: 10 * 10 * 10 nm<sup>3</sup>

* Sample: Human hepatic cells
* Labels region: 
    * cell Membrane
    * nucleus
    * nucleolus
    * lipid droplets

* Sample 
    1. Day 0
        * data_d00_batch01_loc01 (TH-4891-0dayDOX-B-M3)
        * data_d00_batch02_loc02 (TH-4891 0day DOX D-K8)
        * data_d00_batch02_loc03 (TH-4891 0day DOX D A12)
    2. Day 7 
        * data_d07_batch01_loc01 (TH-4891-7dayDOX)
        * data_d07_batch02_loc01 (TH-4891 7day DOX B-L8)
        * data_d07_batch02_loc02 (TH-4891 7day DOX C-R19)
    3. Day 14
        * data_d14_batch01_loc01 (TH-4891-14dayDOX-C-H16)
    4. Day 17
        * data_d17_batch01_loc01 (TH-4891-17dayDOX)
    5. Day 21
        * data_d21_batch01_loc01 (TH-4891-21dayDOX)
    
### Directory
* Data Location: Workstation-008
* 
* Export Directory: `D:\PerlmutterData\segmentation_compiled_export`
    1. `Amira_3D`: Raw Segementation files and Amira .hx files
        * Raw files
            * `raw_img`: raw image files; pixel size 10 nm^3
            * `cell_membrane`: segmentation of area inside cell membrane (WUCCI-DL)
            * `nucleus`: segmentation of nucleus (WUCCI-DL)
            * `nucleolus`: segmentation of nucleolus (Amira)
            * `lipid_droplet`: segmentation of lipid droplets (WUCCI-DL)
            * `mito`: segmentation of mitochondria (Ariadne)
            * `cristae`: segmentation of cristae inside mitochondria (Ariadne)
            * `ER`: segmentation of endoplasmic reticulum (Ariadne)
            * `inclusion`: segmentation of globulus inclusion (Ariadne)
        * Amira .hx file and folders 
            * `data_d**_batch**_loc**_inclusion`: label analysis on ER and inclusion 
            * `data_d**_batch**_loc**_mito`: label analysis on mito and cristae
            * `data_d**_batch**_loc**_skeleton`: skeleton (branching) analysis on mitochondira
        * `.hxtemplate`: template files used for analysis 
    2. `data`: data  
            * `raw`: raw analysis exported from Amira
                * label analysis
                    * `cell_membrane`
                    * `nucleus`
                    * `mito`
                    * `cristae`
                    * `ER`
                    * `inclusion`
                * skeleton/branching analysis
                    * `skeleton`: .xml files; includes node, points and segments
                    * `skeleton_ouput`: .csv files; includes node, points, segments and segments_s
            * `compile`: complied data in single spreadsheet
                * label analysis: `Volume3d`, `Area3d`, `BaryCenterX`, `BaryCenterY`, `BaryCenterZ`, `index`, `filename`, `day`
                    * `cell_membrane`: 
                    * `nucleus`: 
                    * `nucleolus`: 
                    * `lipid_droplet`: 
                    * `mito`: 
                    * `cristae`: 
                    * `ER`: 
                    * `inclusion`: 

    
    
    3. `input`
    4. `output_AND`
    5. `surface`
    
    
    

## Segmentation
* Segmentation has been done in two way: 
    1. WUCCI-DL
    2. Ariadne (https://ariadne.ai/)

## Methods
* Convolutional Neural Network
* U-Net


















## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.