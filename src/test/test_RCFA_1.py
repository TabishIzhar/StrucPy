#Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.

from StrucPy.RCFA import RCF
from StrucPy.RCFA import RCFenv
import pandas as pd
import numpy as np
import pytest
import ray

class Test_RCF_beam:
    
    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('src/test/inputfile_RCFA_beam.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('src/test/inputfile_RCFA_beam.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('src/test/inputfile_RCFA_beam.xlsx', 'boundary', header = 0, index_col=0)
    point_loads= pd.read_excel('src/test/inputfile_RCFA_beam.xlsx', 'point_loads', header = 0, index_col=0)

    # Creating Object
    r1= RCF(nodes_details,member_details,boundcond,point_loads=point_loads,self_weight=False)

    #Pre processing the model
    r1.preP()

    # Performing Analysis
    r1.RCanalysis()

    actual_reactions1= np.array([[0,101.875,0,0,0,67.917],
                                [0,333.125,0,0,0,-30.417]])

    actual_disp1= np.array([[0,0,0,0,0,0],
                            [0,0,0,0,0,0]])

    actual_max_positive_BM=  67.917
    actual_max_negative_BM=  -29.583
    actual_max_positive_SF=  101.875
    actual_max_negative_SF=  -333.125
    max_defl= -0.298

    def test_model3D(self):
        figure= self.r1.model3D()

        # check if it is an empty go.Figure
        if figure.data == tuple():
            status= True
        else:
            status= False
        assert status==False

    def test_GlobalK(self):
        # Unit Test on Global stiffness matrix
        Global_stiffness= self.r1.GlobalK()
        detGK = np.linalg.det(Global_stiffness) 
        assert detGK==pytest.approx(0,0.001)

    def test_LocalK(self):
        # Unit Test on local stiffness matrix
        local_stiffness= self.r1.LocalK()
        detLK = np.linalg.det(local_stiffness) 
        assert detLK==pytest.approx(0,0.001)

    def test_reactions(self):
        # Unit Test on Reactions
        reactions= self.r1.reactions()
        print (((reactions-self.actual_reactions1)==0).all().all())
        assert (((reactions-self.actual_reactions1)==0).all().all())

    def test_Gdisp(self):
        # Unit Test on global nodal displacement
        gdisp= self.r1.Gdisp()
        print
        print ((gdisp==self.actual_disp1).all().all())
        assert ((gdisp==self.actual_disp1).all().all())

    def test_maxmemF(self):
        # Unit Test on maximum member forces
        mf= self.r1.maxmemF()
        assert (mf.max(axis=0)[5]-self.actual_max_positive_BM)==0 and (mf.min(axis=0)[5]-self.actual_max_negative_BM)==0 and (mf.max(axis=0)[1]-self.actual_max_positive_SF)==0 and (mf.min(axis=0)[1]-self.actual_max_negative_SF)==0

    def test_maxdefL(self):
        # Unit Test on maximum deflection in both local and global coordinate system
        defl= self.r1.maxdefL()
        defLD= self.r1.defLD()[0]
        defGD= self.r1.defGD()[0]

        relative_error= abs((defl.min(axis=0)[1]- self.max_defl)/self.max_defl)
        assert (relative_error<=0.05) and (((defGD-defLD)==0).all()== True)


class Test_RCF_2D_Frame:
    
    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('src/test/inputfile_RCFA_2D_Frame.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('src/test/inputfile_RCFA_2D_Frame.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('src/test/inputfile_RCFA_2D_Frame.xlsx', 'boundary', header = 0, index_col=0)
    forcesnodal= pd.read_excel('src/test/inputfile_RCFA_2D_Frame.xlsx', 'forcevec', header = 0, index_col=0)

    # Creating Object
    r2= RCF(nodes_details,member_details,boundcond,forcesnodal=forcesnodal,self_weight=True)

    #Pre processing the model
    r2.preP()

    # Performing Analysis
    r2.RCanalysis()

    actual_reactions2= np.array([[-365.922,31.368,0,0,0,819.652],
                                [-134.078,503.507,0,0,0,0]])

    actual_disp2= np.array([[0,0,0,0,0,0],
                            [159.068,-0.048,0,0,0,-0.021],
                            [158.869,-0.887,0,0,0,0.003],
                            [0,0,0,0,0,-0.061]])

    actual_mem_forces=np.array([[31.368,365.922,0.0,0.0,0.0,819.652],
                                [0,	0, 0.0,	0.0,	0.0, -644.036],
                                [134.078,22.368,0.0,0.0,0.0,536.312],
                                [0, -494.507, 0.0, 0.0, 0.0, -644.383],
                                [503.507, 134.078, 0.0, 0.0, 0.0, 0.0],
                                [0, 0, 0.0, 0.0, 0.0, -536.312]])

    actual_mem_delf_local=np.array([[0, 18.647, 0],
                                    [0,	-4.927, 0],
                                    [0, 0, 0],
                                    [0, -19.053, 0],
                                    [0, 0, 0],
                                    [0, -32.615, 0]])
    def test_model3D(self):
        figure= self.r2.model3D()

        # check if it is an empty go.Figure
        if figure.data == tuple():
            status= True
        else:
            status= False
        assert status==False

    def test_GlobalK(self):
        # Unit Test on Global stiffness matrix
        Global_stiffness= self.r2.GlobalK()
        detGK = np.linalg.det(Global_stiffness) 
        assert detGK==pytest.approx(0,0.001)

    def test_LocalK(self):
        # Unit Test on local stiffness matrix
        local_stiffness= self.r2.LocalK()
        count = 0
        for i in range (self.r2.tm):
            detLK = np.linalg.det(local_stiffness[i,:,:]) 
            if detLK==0:
                count= count +1
        assert (count==self.r2.tm)

    def test_reactions(self):
        # Unit Test on Reactions
        reactions= self.r2.reactions()
        relative_error= abs((reactions-self.actual_reactions2)/self.actual_reactions2)
        relative_error = relative_error.fillna(0)
        assert (relative_error<= 0.01).all().all()== True

    def test_Gdisp(self):
        # Unit Test on global nodal displacement
        gdisp= self.r2.Gdisp()
        relative_error= abs((gdisp-self.actual_disp2)/self.actual_disp2)
        relative_error = relative_error.fillna(0)
        assert (relative_error<= 0.1).all().all()== True

    def test_maxmemF(self):
        # Unit Test on maximum member forces
        mf= self.r2.maxmemF()

        relative_error= abs((mf-self.actual_mem_forces)/self.actual_mem_forces)
        relative_error = relative_error.fillna(0)
        assert (relative_error<= 0.01).all().all()== True


    def test_maxdefL(self):
        # Unit Test on Reactions
        defl= self.r2.maxdefL()
        relative_error= abs((defl-self.actual_mem_delf_local)/self.actual_mem_delf_local)
        relative_error = relative_error.fillna(0)
        assert (relative_error<=0.1).all().all()== True


class Test_RCF_3D_Frame:
    
    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    framegen=  pd.read_excel('src/test/inputfile_RCFA_3D_Frame.xlsx', 'framegen', header = 0, index_col=0)
    seismic_defination= pd.read_excel('src/test/inputfile_RCFA_3D_Frame.xlsx', 'Seismic_Defination', header = 0, index_col=0)
    load_combo= pd.read_excel('src/test/inputfile_RCFA_3D_Frame.xlsx', 'load_combinations', header = 0, index_col=0)

    # Creating RC frame object for analysis
    r3= RCF(nodes_details = None,member_details= None,boundarycondition= None,framegen= framegen,seismic_def=seismic_defination, load_combo= load_combo,  autoflooring= True) 

    #Pre processing the model
    r3.preP()

    r3.changeFrame(member= 'all',width= 400, depth= 400)
    r3.preP()
    r3.changeFrame(member= 'beam', yudl= -10)

    r3.preP()
    r3.changeFL(thickness= 100, LL= -3 , FF=-5, WP=0)
    
    # Performing Analysis
    r3.RCanalysis()

    actual_reactions3= np.array([[-70.73, 521.331, 11.481, 19.029, -0.088, 229.763],
                                [-68.87, 1325.513, 0, 0, 0, 227.129],
                                [-70.73, 521.33, -11.481, -19.029, 0.088, 229.763],
                                [-97.121,1755.557,16.813,27.869,-0.037,273.432],
                                [-97.169,2962.153,0,0,0,273.961],
                                [-97.121,1755.557,-16.813,-27.869,0.037,273.432],
                                [-95.399,1755.581,17.154,28.433,-0.043,270.627],
                                [-95.763,2999.098,0,0,0,271.693],
                                [-95.399,1755.581,-17.154,-28.433,0.043,270.627],
                                [-97.748,1661.837,16.812,27.866,-0.027,274.516],
                                [-98.428,2867.75,0,0,0,276.116],
                                [-97.748,1661.837,-16.812,-27.866,0.027,274.516],
                                [-80.083,1733.297,11.495,19.055,-0.077,245.302],
                                [-82.497,2551.885,0,0,0,249.774],
                                [-80.083,1733.297,-11.495,-19.055,0.077,245.302]])



    actual_disp3= np.array([[0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [26.465,-0.637,-0.014,0.001,0,-0.005],
                            [26.568,-1.642,0,0,0,-0.005],
                            [26.465,-0.637,0.014,-0.001,0,-0.005],
                            [26.482,-2.179,-0.022,0.001,0,-0.003],
                            [26.587,-3.688,0,0,0,-0.003],
                            [26.482,-2.179,0.022,-0.001,0,-0.003],
                            [26.491,-2.179,-0.022,0.001,0,-0.003],
                            [26.6,-3.734,0,0,0,-0.003],
                            [26.491,-2.179,0.022,-0.001,0,-0.003],
                            [26.493,-2.062,-0.022,0.001,0,-0.003],
                            [26.605,-3.57,0,0,0,-0.003],
                            [26.493,-2.062,0.022,-0.001,0,-0.003],
                            [26.486,-2.152,-0.015,0.001,0,-0.004],
                            [26.599,-3.175,0,0,0,-0.004],
                            [26.486,-2.152,0.015,-0.001,0,-0.004],
                            [62.224,-1.222,0.002,0.001,0,-0.005],
                            [62.562,-3.055,0,0,0,-0.005],
                            [62.224,-1.222,-0.002,-0.001,0,-0.005],
                            [62.225,-3.973,0.001,0.001,0,-0.003],
                            [62.564,-6.712,0,0,0,-0.003],
                            [62.225,-3.973,-0.001,-0.001,0,-0.003],
                            [62.228,-4.002,0.001,0.001,0,-0.003],
                            [62.568,-6.832,0,0,0,-0.003],
                            [62.228,-4.002,-0.001,-0.001,0,-0.003],
                            [62.229,-3.795,0.001,0.001,0,-0.003],
                            [62.571,-6.532,0,0,0,-0.003],
                            [62.229,-3.795,-0.001,-0.001,0,-0.003],
                            [62.23,-3.909,-0.001,0.001,0,-0.004],
                            [62.572,-5.778,0,0,0,-0.004],
                            [62.23,-3.909,0.001,-0.001,0,-0.004],
                            [97.163,-1.749,0.001,0.001,0,-0.005],
                            [97.865,-4.244,0,0,0,-0.005],
                            [97.163,-1.749,-0.001,-0.001,0,-0.005],
                            [97.164,-5.39,-0.004,0.001,0,-0.003],
                            [97.868,-9.103,0,0,0,-0.003],
                            [97.164,-5.39,0.004,-0.001,0,-0.003],
                            [97.167,-5.461,-0.003,0.001,0,-0.003],
                            [97.871,-9.304,0,0,0,-0.003],
                            [97.167,-5.461,0.003,-0.001,0,-0.003],
                            [97.168,-5.189,-0.002,0.001,0,-0.003],
                            [97.874,-8.9,0,0,0,-0.003],
                            [97.168,-5.189,0.002,-0.001,0,-0.003],
                            [97.169,-5.261,-0.005,0.001,0,-0.004],
                            [97.875,-7.804,0,0,0,-0.004],
                            [97.169,-5.261,0.005,-0.001,0,-0.004],
                            [128.539,-2.198,0.006,0.001,0,-0.004],
                            [129.742,-5.19,0,0,0,-0.005],
                            [128.539,-2.198,-0.006,-0.001,0,-0.004],
                            [128.538,-6.436,0.001,0.001,0,-0.003],
                            [129.743,-10.879,0,0,0,-0.003],
                            [128.538,-6.436,-0.001,-0.001,0,-0.003],
                            [128.539,-6.552,0.002,0.001,0,-0.003],
                            [129.745,-11.156,0,0,0,-0.003],
                            [128.539,-6.552,-0.002,-0.001,0,-0.003],
                            [128.54,-6.24,0.004,0.001,0,-0.002],
                            [129.746,-10.68,0,0,0,-0.002],
                            [128.54,-6.24,-0.004,-0.001,0,-0.002],
                            [128.541,-6.215,-0.003,0.001,0,-0.003],
                            [129.746,-9.267,0,0,0,-0.003],
                            [128.541,-6.215,0.003,-0.001,0,-0.003],
                            [153.361,-2.537,-0.009,0.001,0,-0.003],
                            [155.169,-5.865,0,0,0,-0.004],
                            [153.361,-2.537,0.009,-0.001,0,-0.003],
                            [153.366,-7.12,-0.023,0.001,0,-0.002],
                            [155.178,-12.055,0,0,0,-0.002],
                            [153.366,-7.12,0.023,-0.001,0,-0.002],
                            [153.372,-7.273,-0.021,0.001,0,-0.002],
                            [155.187,-12.39,0,0,0,-0.002],
                            [153.372,-7.273,0.021,-0.001,0,-0.002],
                            [153.376,-6.942,-0.018,0.001,0,-0.002],
                            [155.192,-11.874,0,0,0,-0.002],
                            [153.376,-6.942,0.018,-0.001,0,-0.002],
                            [153.38,-6.797,-0.022,0.001,0,-0.002],
                            [155.196,-10.192,0,0,0,-0.002],
                            [153.38,-6.797,0.022,-0.001,0,-0.002],
                            [168.508,-2.716,0.055,0.002,0,-0.002],
                            [170.901,-6.226,0,0,0,-0.003],
                            [168.508,-2.716,-0.055,-0.002,0,-0.002],
                            [168.487,-7.45,0.064,0.002,0,-0.001],
                            [170.874,-12.653,0,0,0,-0.001],
                            [168.487,-7.45,-0.064,-0.002,0,-0.001],
                            [168.463,-7.62,0.069,0.003,0,-0.001],
                            [170.841,-13.015,0,0,0,-0.001],
                            [168.463,-7.62,-0.069,-0.003,0,-0.001],
                            [168.445,-7.285,0.07,0.002,0,-0.001],
                            [170.811,-12.485,0,0,0,-0.001],
                            [168.445,-7.285,-0.07,-0.002,0,-0.001],
                            [168.434,-7.044,0.038,0.002,0,-0.001],
                            [170.791,-10.623,0,0,0,0],
                            [168.434,-7.044,-0.038,-0.002,0,-0.001]])

    actual_seismic_detail= np.array([[0.16,1.5,3,1.415,5,2,0.96140,0.0566,19512.00]])
    
    actual_seismic_shear= np.array([[1104.07],[12.358],[49.433],[111.224],[197.731],[308.955],[424.374]])

    actual_story_disp= np.array([0,26.52,62.341,97.401,128.941,153.976,169.259])

    actual_mem_forces_19=np.array([[0,0,0.242,0,0.397,290.494],
                                [-16.889,-196.273,0,0.0,-0.573, -259.399]])
    actual_mem_forces_220=np.array([[775.437,63.572,0.0,0.0, 55.114,151.884],
                                    [0,0 ,-22.274,0.0,-56.256,-165.975]])

    actual_mem_delf_local=np.array([[0, 18.647, 0],
                                    [0,	-4.927, 0],
                                    [0, 0, 0],
                                    [0, -19.053, 0],
                                    [0, 0, 0],
                                    [0, -32.615, 0]])
    def test_model3D(self):
        figure= self.r3.model3D()

        # check if it is an empty go.Figure
        if figure.data == tuple():
            status= True
        else:
            status= False
        assert status==False

    def test_LocalK(self):
        # Unit Test on local stiffness matrix
        local_stiffness= (self.r3.LocalK())
        count = 0
        for i in range (self.r3.tm):
            detLK = np.linalg.det(local_stiffness[i,:,:]) 
            if detLK==0:
                count= count +1
        assert (count==self.r3.tm)

    @pytest.mark.skipif(not r3.seismic_def_status, reason="No Seismic Anlaysis Performed")
    def test_seismicD(self):
        # Unit Test on Seismic Data such as Time Period etc. 
        seismic_detail= self.r3.seismicD()
        relative_error= abs((seismic_detail-self.actual_seismic_detail)/self.actual_seismic_detail)
        relative_error = relative_error.fillna(0)
        assert (relative_error<= 0.001).all().all()== True


    @pytest.mark.skipif(not r3.seismic_def_status, reason="No Seismic Anlaysis Performed")
    def test_seismicS(self):
        # Unit Test on Seismic Shear
        seismic_shear= self.r3.seismicS()
        relative_error= abs((seismic_shear-self.actual_seismic_shear)/self.actual_seismic_shear)
        relative_error = relative_error.fillna(0)
        assert (relative_error<= 0.001).all().all()== True

    @pytest.mark.skipif(not r3.seismic_def_status, reason="No Seismic Anlaysis Performed")
    def test_Sdrift(self):
        # Unit Test on Story Displacement
        story_drift= self.r3.Sdrift()
        story_disp=story_drift.iloc[:,0]
        relative_error= abs((story_disp-self.actual_story_disp)/self.actual_story_disp)
        relative_error = relative_error.fillna(0)
        assert (relative_error<= 0.1).all().all()== True

    def test_reactions(self):
        # Unit Test on Reactions
        reactions= self.r3.reactions()
        relative_error= abs((reactions-self.actual_reactions3)/self.actual_reactions3)
        relative_error = relative_error.fillna(0)
        assert (relative_error<= 1).all().all()== True

    def test_Gdisp(self):
        # Unit Test on global nodal displacement
        gdisp= self.r3.Gdisp()
        error= abs((gdisp-self.actual_disp3))
        error = error.fillna(0)
        assert (abs(error)<= 5).all().all()== True

    def test_maxmemF(self):
        # Unit Test on maximum member forces
        mf= self.r3.maxmemF()
        mf19= mf.loc[19]
        mf220= mf.loc[220]

        relative_error19= abs((mf19-self.actual_mem_forces_19)/self.actual_mem_forces_19)
        relative_error220= abs((mf220-self.actual_mem_forces_220)/self.actual_mem_forces_220)

        relative_error19 = relative_error19.fillna(0)
        relative_error220 = relative_error220.fillna(0)

        relative_error19.replace([np.inf, -np.inf], 0, inplace=True)
        relative_error220.replace([np.inf, -np.inf], 0, inplace=True)

        assert ((relative_error19<= 1).all().all()== True) and ((relative_error220<= 1).all().all()== True)


  
class Test_RCFenv_3D_Frame:
    
    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    framegen=  pd.read_excel('src/test/inputfile_RCFA_RCFenv_3D_Frame.xlsx', 'framegen', header = 0, index_col=0)
    seismic_defination= pd.read_excel('src/test/inputfile_RCFA_RCFenv_3D_Frame.xlsx', 'Seismic_Defination', header = 0, index_col=0)
    load_combo= pd.read_excel('src/test/inputfile_RCFA_RCFenv_3D_Frame.xlsx', 'load_combinations', header = 0, index_col=0)


    # Creating RC frame object for analysis
    r4= RCF(nodes_details = None,member_details= None,boundarycondition= None,framegen= framegen,seismic_def=seismic_defination, load_combo= load_combo,  autoflooring= True) 
    #Pre processing the model
    r4.preP()

    mem_list = [121,122,123,125,126,127,129,130,131,111,112,113,114,115,116,138,144,150,156,162,168,174,180,186,99,100,103,104,107,108,89,90,91,92,137,143,149,155,161,167,67,68,77,81,85,136,148,142]
    r4.changeFrame(member= mem_list, delete= True)

    r4.preP()

    r4.changeFrame(member= 'all',width= 400, depth= 400)
    r4.preP()
    r4.changeFrame(member= 'beam', yudl= -10)

    r4.preP()
    r4.changeFL(thickness= 100, LL= -3 , FF=-5, WP=0)
    
    member_details= r4.modelMD()
    nodes_details= r4.modelND()
    boundary_conditions= r4.modelBCD()
    floor_detail= r4.floorD()
    
    member_details.drop(['Type'], axis=1,inplace=True)
    floor_detail.drop(['Floor'], axis=1,inplace=True)

    # Creating RC frame object for structural anlaysis for different load combinations
    r5= RCFenv(nodes_details = nodes_details, member_details= member_details, boundarycondition= boundary_conditions , load_combo= load_combo, seismic_def= seismic_defination, slab_details= floor_detail)

    r5.preP()

    r5.RCanalysis()


    actual_reaction_node1= np.array([48.727, 803.606, 40.882, 118.841, 7.242, -152.389])

    actual_reaction_node11= np.array([-62.456, 2882.971, 80.030, 235.345, 6.843, 175.638])                               

    actual_disp_node31= np.array([39.313, -1.631, -32.674, 0.003, 0.002, 0.003])

    actual_disp_node59= np.array([60.991, -7.505, 108.13, 0.004, 0.003, 0.003]) 


    def test_getReactmax(self):
        # Unit Test on Reactions
        getReactmax= self.r5.getReactmax()
        getReactmax_node1= getReactmax.loc[1]
        getReactmax_node11= getReactmax.loc[11]

        relative_error_node1= abs((getReactmax_node1.iloc[0,:]-self.actual_reaction_node1)/self.actual_reaction_node1)

        relative_error_node11= abs((getReactmax_node11.iloc[0,:]-self.actual_reaction_node11)/self.actual_reaction_node11)

        relative_error_node1 = relative_error_node1.fillna(0)
        relative_error_node11 = relative_error_node11.fillna(0)

        assert ((relative_error_node1<=1 ).all().all()== True) and ((relative_error_node11<=1 ).all().all()== True)

    def test_getNdismax(self):
        # Unit Test on global nodal displacement
        getNdismax= self.r5.getNdismax()
        getNdismax_node31= getNdismax.loc[31]
        getNdismax_node59= getNdismax.loc[59]
    
        error_node31= abs((getNdismax_node31.iloc[0,:]-self.actual_disp_node31))

        error_node59= abs((getNdismax_node59.iloc[0,:]-self.actual_disp_node59))

        error_node31 = error_node31.fillna(0)
        error_node59 = error_node59.fillna(0)

        assert ((abs(error_node31)<= 2).all().all()== True) and ((abs(error_node59)<= 2).all().all()== True)