# MachineLearnModel

<p>This project contains scripts file of feature generation, feature screening, model training and transfer integral prediction required for machine learning to predict four molecular transfer integrals</p>
<p><strong>########## Software/Libraries Requirement ##########</strong></p>
<ul>
<li>Python 3.7</li>
<li>Scikit Learn version 0.24.1</li>
<li>Numpy version 1.16.2</li>
<li>PyTorch version 1.8.0</li>
</ul>

<p>Folder qt、pen、rub、dntt contain data and scripts required for model training and preditiction for quadruple thiophene, pentacene, rubrene, and dinaphtho[2,3-b:2',3'-f]thieno[3,2-b]thiophene </p>
<p>Each folder contains the following files ：</p>
<ul>
<li>data0.txt #XYZ coordinates of each atom extracted from MD simulations, unit in nm</li>
<li>ce0 #cell vectors of periodic bounding box, arranged by XX YY ZZ XY XZ YX YZ ZX ZY, unit in nm</li>
<li>id0 #near neighbor list file, serial number starting from 1</li>
<li>elelist #element list of the molecule, new structure needs to be generated in this order to ensure that scripts read the atomic order correctly.</li>
<li>import.txt #Importance ranking obtained using the feature filter method</li>
<li>make321mutil.py #Scripts for generating overlap matrices, the parameters for element specificity already included in the script. Overlap matrices element of hydrogen atoms or intra-molecules will not created.</li>
<li>jefflod0.txt #Effective transfer integrals obtained by quantum chemical calculations and lowdin orthogonalization. Units in hartree and non-physical phase has not benn corrected.</li>
<li>makefilter.py #Scripts for feature filtering</li>
<li>torjesa.py #Neural network training program, the trained model will be saved as mlp.pth, which can be called by the torch.load , the phase and unit conversion of the labels and the batch normalization of the features are done in this script. Example features are the feature filtered overlap matrix , and the best performing hyperparameters are listed in the script.</li>
<li>pred.py #Scripts for transfer integral prediction</li>
</ul>

<p><strong>########## Feature Generation ##########</strong></p>
<ol>
<li>Make sure make321mutil.py, id0, data0.txt and ce0 are all downloaded and placed in the same folder</li>
<li>Run </li>

  ```
  ./make321mutil.py
  ```
 
<li>A feature file A321exx0.txt will be generated</li>
<p><strong>########## Feature Filtering ##########</strong></p>
<ol>
<li>Make sure makefilter.py, A321exx0.txt and import.txt are all downloaded and placed in the same folder</li>
<li>Run </li>

  ```
  ./makefilter.py
  ```
 
<li>Filtered feature file A321exx0_edit1.txt will be generated</li>
</ol>
<p><strong>########## Model Training ##########</strong></p>
<ol>
<li>Modify the python interpreter path to your own path and make sure you have the right version of the dependent ibraries</li>
<li>Run torjesa.py. The performance of the model will printed to the screen during training.</li>
<li>Model file mlp.pth will be generated</li>
</ol>
<p><strong>########## Prediction with Models ##########</strong></p>
<ol>
<li>Prepare your own 3D coordinate files, lattice vector files and near neighbor list files, making sure that the atomic order and file format are exactly the same as those provided in the project</li>
<li>Modify make321mutil.py and makefilter and run to obtain features of your own structure</li>
<li>Modify pred.py and run</li>
<li>Predict file will be generated</li>
</ol>

<p>Folder Ehanced contain data and scripts required for data augmentation，quadruple thiophene is used as an example </p>
<ul>
<li>data0.txt #XYZ coordinates of each atom extracted from MD simulations, unit in nm</li>
<li>ce0 #cell vectors of periodic bounding box, arranged by XX YY ZZ XY XZ YX YZ ZX ZY, unit in nm</li>
<li>id0 #near neighbor list file, serial number starting from 1</li>
<li>datahigh1.txt #XYZ coordinates of each atom extracted from 400K MD simulations, unit in nm</li>
<li>ce1 #cell vectors of periodic bounding box extracted from 400K MD simulations, arranged by XX YY ZZ XY XZ YX YZ ZX ZY, unit in nm</li>
<li>idf1 #near neighbor list of 8181 dimers in edge region of the original dataset , serial number starting from 1</li>
<li>elelist #element list of the molecule, new structure needs to be generated in this order to ensure that scripts read the atomic order correctly.</li>
<li>import.txt #Importance ranking obtained using the feature filter method</li>
<li>make321mutil.py #Scripts for generating overlap matrices, the parameters for element specificity already included in the script. Overlap matrices element of hydrogen atoms or intra-molecules will not created.</li>
<li>make321mutil.py #Scripts for generating rotation augmentation overlap matrices,.</li>
<li>jefflod0.txt #Effective transfer integrals obtained by quantum chemical calculations and lowdin orthogonalization. Units in hartree and non-physical phase has not benn corrected.</li>
<li>jefflodh1.txt #Effective transfer integrals of 8181 dimers in edge region.</li>
<li>makefilter.py #Scripts for feature filtering</li>
<li>torjesarot.py #Neural network training program for rotation augmented dataset.</li>
<li>torlong.py #Neural network training program for enhanced sampling dataset.</li>
<li>pred.py #Scripts for transfer integral prediction</li>
</ul>

<p><strong>########## Rotation Augmentation  ##########</strong></p>
<ol>
<li>Run make321rot.py, Rotation Augmentation feature A321exxrot.txt will be genreated.</li>
<li>Modify makefilter.py and run,Filtered feature file A321rot_edit1.txt will be generated </li>
<li>Run torjesarot.py,model file mlprot.pth will be generated</li>
</ol>
<p><strong>########## Enhanced Sampling  ##########</strong></p>
<ol>
<li>Run make321mutil.py, feature of 8181 dimers in edge region of the original dataset, A321exxh1.txt will be genreated.</li>
<li>Modify makefilter.py and run,Filtered feature file A321h1_edit1.txt will be generated </li>
<li>Copy ../qt/A3210_edit1.txt to current folder </li>
<li>run "cat A3210_edit1.txt A321h1_edit1.txt > A3210h1_edit1.txt" and "cat jefflod0.txt jefflodh1.txt > jefflodh1.txt"</li>
<li>Copy ../qt/mlp.pth to current folder as a pre-training models</li>
<li>Run torlong.py,model file mlplong.pth will be generated</li>
</ol>
<p><strong>########## Prediction with Models ##########</strong></p>
The steps are exactly the same as without  data augmentation

The pre-trained model can be downloaded in https://mega.nz/folder/OmBDRCga#NcbZ8E6wVXcOKVs1TlDFhg, and use
```
torch.load(PATH)
```
to load model
