import numpy as np
import pandas as pd
import plotly.graph_objects as go 
import itertools as itr  
from plotly.subplots import make_subplots     
import copy
import ray
from ._RCSingle import _RCFforenvelop
from ._arrangeCal import *
from ._forcesCal import *

class RCF():
    """
    This is a class to represent the 2D/3D reinforced concrete members/ frame model.
    It creates an object for the reinforced concrete members/ frames model on which analysis can be performed. 

    :param nodes_details: A handle to the :class:`StrucPy.RCFA.RCF` that detects the nodes and their coordinates in space. Check :ref:`InputExample:Nodes Details` for more details.
    :type nodes_details: DataFrame
    
    :param member_details: A handle to the :class:`StrucPy.RCFA.RCF` that detects the members of reinforced concrete frame along with thier nodes, cross-section and loading details. Check :ref:`InputExample:Member Details` for more details.
    :type member_details: DataFrame
    
    :param boundcondition: A handle to the :class:`StrucPy.RCFA.RCF` that detects the nodes/joints condition of reinforced concrete frame i.e. types of supports (fixed, hinged etc.) and joints conditions. Check :ref:`InputExample:Boundary Conditions` for more details.
    :type boundcondition: DataFrame

    :param framegen: A handle to the :class:`StrucPy.RCFA.RCF` that detects the number of bays and total length of required reinforced concrete frame along x- axis, z-axis and height along y-axis. Check :ref:`InputExample:framegen` for more details.
    :type framegen: DataFrame    

    :param forcesnodal: A handle to the :class:`StrucPy.RCFA.RCF` that detects the nodes/joints forces of reinforced concrete frame, defaults to None (No nodel forces or moments). Check :ref:`InputExample:Nodal Forces Details` for more details.
    :type forcesnodal: DataFrame, optional
    
    :param slab_details:  A handle to the :class:`StrucPy.RCFA.RCF` that detects the slabs/floor loads along with the nodes within which a slab/floor is formed, defaults to None (No floor load). Check :ref:`InputExample:Slab Details` for more details.
    :type slab_details: DataFrame, optional
    
    :param load_combo:  A handle to the :class:`StrucPy.RCFA.RCF` that detects the load combination for the analysis of reinforced concrete frame. It takes the load factor respective to load case. Defaults to None (Only Dead Load case will be considered with load factor 1). Check :ref:`InputExample:Load Combination Details` for more details.
    :type load_combo: DataFrame, optional
    
    :param seismic_def:  A handle to the :class:`StrucPy.RCFA.RCF` that detects the seismic load being applied to reinforced concrete frame. Defaults to None (No Seismic Load). Check :ref:`InputExample:Seismic Definition` for more details.
    :type seismic_def: DataFrame, optional
    
    :param properties: A handle to the :class:`StrucPy.RCFA.RCF` that detects the properties of the members to be considered in analysis. Default to None (Concrete of Grade M25 ). Check :ref:`InputExample:Material Properties` for more details.
    :type properties: DataFrame, optional
    
    :param grade_conc: A handle to the :class:`StrucPy.RCFA.RCF` that detects the grade of concrete (like M25, M30) to be considered in analysis. Defaults to 25 N/mm2. Check :ref:`InputExample:Concrete Grade` for more details.
    :type grade_conc: float/int, optional
    
    :param self_weight:  A handle to the :class:`StrucPy.RCFA.RCF` that detects whether self weight of the members are to be considered in analysis or not. Defaults to True (Self weight considered). Check :ref:`InputExample:Self Weight` for more details.
    :type self_weight: Boolean, optional
    
    :param infillwall:  A handle to the :class:`StrucPy.RCFA.RCF` that detects whether infillwall has to be considered during the caculation of time period during seismic analysis (Applicable only for IS 1893: 2016 Part 1). Defaults to False (infillwall not considered). Check :ref:`InputExample:infillwall` for more details.
    :type infillwall: Boolean, optional
    
    :param autoflooring:  A handle to the :class:`StrucPy.RCFA.RCF` that detects whether the slab/floor load has to be generated automatically or not. It is highly usefull when dealing with large reinforced concrete framed structures. Defaults to False (Autoflooring not being done).  Check :ref:`InputExample:Autoflooring` for more details.
    :type autoflooring: Boolean, optional

    :param point_loads:  A handle to the :class:`StrucPy.RCFA.RCF` that detects the point loads on a members. A member can have multiple point loads. It must be passed as a DataFrame. Index must the ID(name) of a member along with three columns. Defaults to None (No point loads). Check :ref:`InputExample:Point Loads` for more details.
    :type point_loads: DataFrame, optional

    :param col_stablity_index:  A handle to the :class:`StrucPy.RCFA.RCF` that determines whether the column or frame is sway or not. It is highly usefull when dealing with large reinforced concrete framed structures. Defaults to 0.04 (IS456:2000).  Check :ref:`InputExample:Stability Index` for more details.
    :type col_stablity_index: Float/Int, optional
    """
        
    def __init__(self, nodes_details, member_details, boundarycondition, framegen= None,forcesnodal=None, slab_details=None, load_combo=None, seismic_def=None,self_weight= True, infillwall=False, autoflooring= False, properties= None, grade_conc= 25, point_loads= None, col_stablity_index= 0.04):
        
        if framegen is not None:
            if not isinstance(framegen, pd.DataFrame):
                raise TypeError ("Type of 'framegen' must be DataFrame")    

            if len(framegen.columns) != 2:
                raise Exception ("framegen must have 2 columns: ['Number of bays', 'Total Length']")
            
            if len(framegen.index) != 3:
                raise Exception ("framegen must have 3 rows: ['Along length (x-axis) ', 'Along height (y-axis)'], 'Along width (z-axis)']")            

            lx= framegen.iat[0,1]
            ly= framegen.iat[1,1]
            lz= framegen.iat[2,1]
            nx= framegen.iat[0,0]
            ny= framegen.iat[1,0]
            nz= framegen.iat[2,0]
            x = np.linspace(0, lx, nx+1)
            y = np.linspace(0, ly, ny+1)
            z = np.linspace(0, lz, nz+1)


            zv,xv = np.meshgrid(z,x)

            xv= xv.flatten()
            zv= zv.flatten()

            zvv, yv = np.meshgrid(zv, y)

            xvv,yvv=  np.meshgrid( xv, y)

            cord_array= np.vstack((xvv.flatten(), yv.flatten(), zvv.flatten())).T
            nodes_details= pd.DataFrame(cord_array, columns= ['x','y','z'], index = [i for i in range (1,len (cord_array)+1)])


            total_members= (((nx* (nz+1))+ ((nx+1)* nz)) * ny) + (((nx+1)* (nz+1) * ny))
            mem_cords= np.empty([total_members,2])
            member_number= 0

            for ka in range (1,len(y)):
                y_val= y[ka]
                new_nodes1= nodes_details[nodes_details.y.isin([y_val])]
                for ia in x:
                    new_node2= new_nodes1[new_nodes1.x.isin([ia])]
                    node_ids= new_node2.index.to_list()
                    for ja in range (len(new_node2)-1):
                        mem_cords[member_number, 0] = node_ids[ja]
                        mem_cords[member_number, 1] = node_ids[ja+1]
                        member_number = member_number+1

                for ia in z:
                    new_node2= new_nodes1[new_nodes1.z.isin([ia])]
                    node_ids= new_node2.index.to_list()
                    for ja in range (len(new_node2)-1):
                        mem_cords[member_number, 0] = node_ids[ja]
                        mem_cords[member_number, 1] = node_ids[ja+1]
                        member_number = member_number+1

            nodes_details_usable= nodes_details.sort_values(by=['x', 'z']).copy()
            node_ids= nodes_details_usable.index.to_list()

            for ka in x:
                new_nodes1= nodes_details[nodes_details.x.isin([ka])]
                for ia in z:    
                    new_node2= new_nodes1[new_nodes1.z.isin([ia])]
                    node_ids= new_node2.index.to_list()
                    for ja in range (len(new_node2)-1):
                        mem_cords[member_number, 0] = node_ids[ja]
                        mem_cords[member_number, 1] = node_ids[ja+1]
                        member_number = member_number+1       

            mem_cords= mem_cords.astype(np.int64)
            member_details= pd.DataFrame(mem_cords, columns= ['Node1','Node2'], index = [i for i in range (1,len (mem_cords)+1)])

            member_details[['b', 'd']]= 500
            member_details[['xUDL', 'yUDL', 'zUDL']]= 0
            base_nodes= nodes_details[nodes_details.y.isin([y[0]])]

            boundarycondition_array= np.zeros ([len(base_nodes), 6])
            boundarycondition= pd.DataFrame(boundarycondition_array, columns= ["x","y","z","thetax","thetay","thetaz"], index = base_nodes.index)



        if not all(isinstance(i, pd.DataFrame) for i in [nodes_details, member_details, boundarycondition]):
            raise TypeError ("Type of the argument must be DataFrame")

        __member_columns= ['Node1', 'Node2', 'b', 'd','xUDL', 'yUDL', 'zUDL']
        __nodes_columns= ['x', 'y', 'z']

        if len(member_details.columns)<7:
            raise Exception("MEMBER DETAILS must have 7 columns: ['Node1', 'Node2', 'b', 'd', 'xUDL', 'yUDL', 'zUDL'], First 4 columns are mandotory argument while last three loads can be left empty or with zero ")

        if len(member_details.columns)>7 :
            raise Exception("MEMBER DETAILS can have maximum of 7 columns: ['Node1', 'Node2', 'b', 'd','xUDL', 'yUDL', 'zUDL']")        
        
        if len(nodes_details.columns)!=3 :
            raise Exception("NODE DETAILS must have x,y and z coordinate: ['x', 'y', 'z']")
        
        if nodes_details.index.dtype not in ["int32","int64"]:
            raise TypeError("Node Number(Index) in 'nodes_details' must be 'int' type")

        if member_details.index.dtype not in ["int32","int64"]:
            raise TypeError("Member Numbe(Index) in 'member_details' must be 'int' type")
        
        if nodes_details.index.all()<1:
            raise NameError("Node Number(Index) in 'nodes_details' must be positive integer")

        if member_details.index.all()<1:
            raise NameError("Member Number(Index) in 'member_details' must be positive integer")
         

        member_details.columns= __member_columns 
        nodes_details.columns= __nodes_columns

        self.__nodes_details= nodes_details.sort_index()      #self.joint_details to be used
        
        self.__member_details= member_details.sort_index()  #self.mem_details to be used


        for i in range (len(self.__member_details)):
                n1= self.__member_details.iloc[i,0]
                n2= self.__member_details.iloc[i,1]
                if (self.__nodes_details.loc[n1]>self.__nodes_details.loc[n2]).any():
                    n3= n2
                    n2=n1
                    n1= n3
                self.__member_details.iloc[i,0]= n1
                self.__member_details.iloc[i,1]= n2
        
        self.__member_details.fillna(0,inplace=True)
        self.__nodes_details.fillna(0,inplace=True)

        self.member_list= self.__member_details.index.to_list()
        self.node_list= self.__nodes_details.index.to_list()
        self.tn= self.__nodes_details.shape[0]
        self.tm= self.__member_details.shape[0]

        member_nodes_check= self.__member_details.loc[:,["Node1", "Node2"]].isin(nodes_details.index)

        if member_nodes_check.all().all():
            pass
        else:
            raise Exception ("These nodes present in member details does not exist: ",  member_nodes_check[member_nodes_check["Node1"]==False]["Node1"] , member_nodes_check[member_nodes_check["Node2"]==False]["Node2"]  )
        
        
        depth_check= self.__member_details.index[self.__member_details["d"]>=0].to_list()

        if len(depth_check)!= self.tm:
            raise Exception ("The depth of some members has not been fixed: ")


        if forcesnodal is None:
            self.__forcesnodal = pd.DataFrame(np.zeros([self.tn,6]),index=self.node_list,columns=["Fx","Fy","Fz","Mx","My","Mz"])
        
        if forcesnodal is not None:
            if not isinstance(forcesnodal, pd.DataFrame):
                raise TypeError ("Type of the 'forcesnodal' must be DataFrame")           
            
            self.__fv_index = forcesnodal.index.isin(nodes_details.index)

            if self.__fv_index.all():
                if len(forcesnodal)==self.node_list:
                    self.__forcesnodal= forcesnodal.sort_index()
                    self.__forcesnodal.columns= ["Fx","Fy","Fz","Mx","My","Mz"]
                elif len(forcesnodal)!=self.node_list:           
                    self.__forcesnodal = pd.DataFrame(np.zeros([self.tn,6]),index=self.node_list,columns=["Fx","Fy","Fz","Mx","My","Mz"])
                    forcesnodal.sort_index(inplace=True)
                    self.__forcesnodal.loc[forcesnodal.index]= forcesnodal.loc[:]
            else:
                raise Exception ("These nodes in nodal forces DataFrame does not exist: ",  [i for i, val in enumerate(self.__fv_index) if not val] )

        if len(boundarycondition.columns) != 6:
            raise Exception ("The boundary condition dataframe must contain 6 columns representing each degree of freedom in 3D space i.e. 'Trans x', 'Trans y', 'Trans z', 'Rotation x', 'Rotation y', 'Rotation z'.  ")

        self.__bc_index = boundarycondition.index.isin(nodes_details.index)

        if self.__bc_index.all():
            if len(boundarycondition)==self.node_list:
                self.__boundarycondition= boundarycondition.sort_index()
                self.__boundarycondition.columns= ["x","y","z","thetax","thetay","thetaz"]
            elif len(boundarycondition)!=self.node_list:             
                self.__boundarycondition = pd.DataFrame(np.ones([self.tn,6]),index=self.node_list,columns=["x","y","z","thetax","thetay","thetaz"])
                boundarycondition.sort_index(inplace=True)
                self.__boundarycondition.loc[boundarycondition.index]= boundarycondition.loc[:]
        else:
            raise Exception ("These nodes in boundary condition does not exist: ",  [i for i, val in enumerate(self.__bc_index) if not val] )

        self.autoflooring= autoflooring
        self.__self_weight= self_weight

        if slab_details is None and self.autoflooring== False:
            self.__slabload_there= 0
            self.__slab_details= "Slab/Floor not present in the Frame"
        if slab_details is not None and self.autoflooring== False:
            if not isinstance(slab_details, pd.DataFrame):
                raise TypeError ("Type of the 'slab_details' must be DataFrame")
            self.__slabload_there= 1
            self.__slab_details= slab_details.sort_index()
            self.__slab_details.fillna(0,inplace=True)
            self.__slab_details.columns = ['Node1', 'Node2', 'Node3', 'Node4',	'Thickness(mm)', 'FF(kN/m2)', 'LL(kN/m2)', 'Waterproofing(kN/m2)']

        if slab_details is None and self.autoflooring== True:
            self.__slabload_there= 1

            

        if load_combo is None:
            self.load_combo= pd.DataFrame(np.zeros([1,6]),index=[1],columns=["Dead_Load","Live_Load","EQX","-EQx","EQZ","-EQZ"])
            self.load_combo.iloc[0,0]=1
        if load_combo is not None:
            if not isinstance(load_combo, pd.DataFrame):
                raise TypeError ("Type of the 'load_combo' must be of  type 'DataFrame'")           
            self.load_combo= load_combo
            self.load_combo.columns= ["Dead_Load","Live_Load","EQX","-EQx","EQZ","-EQZ"]
            self.load_combo.fillna(0,inplace=True)

        if seismic_def is None:
            self.__seismic_def_status= False
            self.__seismic_def= "No Seismic Anlaysis Performed"
        else:
            self.__seismic_def_status= True
            if not isinstance(seismic_def,pd.DataFrame): 
                raise TypeError ("Type of the 'seismic_def' must be DataFrame")
            self.__seismic_def= seismic_def
            self.__seismic_def.columns= ["Z","I","R","Sag","Damping(%)","Soil Type", "Time Period" ]
            if load_combo is None:
                raise Exception ("Load Factor and Direction of Seismic Forces not defined in Load Combination")

        if point_loads is None:
            self.__point_L= False
            self.__point_loads= None
        else:
            if not isinstance(point_loads,pd.DataFrame): 
                raise TypeError ("Type of the 'point_loads' must be DataFrame")
            self.__point_L= True
            self.__point_loads = point_loads
        

        self.__grade_conc=grade_conc
        E0= 5000*np.sqrt(self.__grade_conc)*(10**3)
        alpha0= 10*(10**(-6))
        mu0= 0.17           
        G0= E0/(2*(1+mu0))
        self.__concrete_density= 25
        self.__concrete_density_beam= 25
        self.__concrete_density_col= 25
        self.__concrete_densitySlab= 25

        self.__DefaultMproperties= pd.DataFrame({ "Type": "All", "Material": "Concrete", "Grade M-": self.__grade_conc, "Density (kN/m3)": self.__concrete_density, "Young Modulus (kN/m2)": E0, "Poisson's Ratio (mu)": mu0, "Thermal Coefficient (alpha)": alpha0, "Critical Damping": 0.05, "Modulus of Rigidity (kN/m2)": G0}, index=[1])

        if properties is None:
            self.__Mproperties= self.__DefaultMproperties.copy()

        if properties is not None:
            if not isinstance(properties, pd.DataFrame):
                raise TypeError ("Type of the 'Material Properties' must be of  type 'DataFrame'")

            typelist = ['all', 'beam', 'column', 'slab']

            properties.columns= ["Type", "Material", "Grade M-", "Density (kN/m3)", "Young Modulus (kN/m2)", "Poisson's Ratio (mu)", "Thermal Coefficient (alpha)", "Critical Damping", "Modulus of Rigidity (kN/m2)"]

            for kk in range (len(properties.index)):
                if not isinstance(properties.iat[kk,0], str):
                    raise Exception ("Something wrong with the 'Type' in Material Properties. It can only be 'All', 'Beam', 'Column' or 'Slab'. ")
            
            if (properties['Type'].str.lower().isin(typelist)).all()==True:
                TypeM= properties['Type'].to_list()
            else:
                raise Exception ("Something wrong with the 'Type' in Material Properties. It can be only 'All', 'Beam', 'Column' or 'Slab'")
            
            for kk in range (len(properties.index)):
                if not isinstance(properties.iat[kk,1], str):
                    raise Exception ("Something wrong with the name of the material in Material Properties")
            NameM= properties['Material'].to_list()

            for kk in range (len(properties.index)):
                if isinstance(properties.iat[kk,2].item(), (float, int)):
                    pass
                else:
                    raise Exception ("Something wrong with the grade of concrete in Material Properties- It must be Int or float number like 20, 25 ,30, 37.5 etc. representing the M20, M25, M30 respectively. ")

            for kk in range (len(properties.index)):
                if properties.iloc[kk,2] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the grade of concrete in second column of Material Properties. It must be positive number.")
            gradeM= properties['Grade M-'].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,3] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of density in second column of Material Properties. It must be positive number.")
            densityM= properties['Density (kN/m3)'].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,4] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Young Modulus in third column of Material Properties. It must be positive number.")
            EM= properties['Young Modulus (kN/m2)'].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,5] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Poisson's Ratio in fourth column of Material Properties. It must be positive number.") 
            muM= properties["Poisson's Ratio (mu)"].to_list()

            for kk in range (len(properties.index)):                            
                if properties.iloc[kk,6] != 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Thermal Coefficient in fifth column of Material Properties") 
            alphaM= properties["Thermal Coefficient (alpha)"].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,7] > 0:
                    pass
                elif properties.iloc[kk,7] == 0:
                    properties.iloc[kk,7]= 0.05
                else:
                    raise Exception ("Something wrong with the value of Critical Damping in sixth column of Material Properties") 
            critM= properties["Critical Damping"].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,8] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Modulus of Rigidity in seventh column of Material Properties. It must be positive number.")
            GM= properties["Modulus of Rigidity (kN/m2)"].to_list()

            prop_index= [i for i in range (1,len(properties)+1)]
            
            self.__Mproperties= pd.DataFrame({ "Type": TypeM ,"Material": NameM, "Grade M-": gradeM, "Density (kN/m3)": densityM, "Young Modulus (kN/m2)": EM, "Poisson's Ratio (mu)": muM, "Thermal Coefficient (alpha)": alphaM, "Critical Damping": critM, "Modulus of Rigidity (kN/m2)": GM}, index= prop_index)

        if self.__self_weight== False:
            self.__concrete_density = 0
            self.__concrete_density_beam= 0
            self.__concrete_density_col= 0
            self.__concrete_densitySlab = 0

        if isinstance(col_stablity_index, (float, int)):
            self.__col_stablity_index= col_stablity_index
        else:
            raise TypeError("Column stability index must be number")

        self.__cords_member_order= None
        self.beams_detail = None
        self.columns_detail = None
        self.nodes_detail= None
        self.slab_pd= None


        self.__local_stiffness= np.empty((len(self.__member_details),12,12))
        self.__lnf= np.empty((len(self.__member_details),12,1))
        self.__K_Global= None
        self.__K_Local= None
        self.__global_forces= None
        self.__trans_mat=None
        self.__GForces= None
        self.__GDisplacement= None
        self.__SF_BM= None
        self.__Deflections_local= None
        self.__Deflections_G= None
        self.__deflected_shape= None


        self.__Framex3D= None
        self.__Framey3D= None
        self.__Framez3D= None

        self.__llvdl= None
        self.__dlvdl= None
        self.__member_nodes_inorder= None

        self.__beam_loads= None
        self.__column_loads= None
        self.story_lumploads= None
        self.__infillwall= infillwall
        
        self.__SeismicShear= None
        self.__floor_loads= None

        self.__PreP_status= False
        self.__Analysis_performed= False
        self.baseN= None
        self.__bd_LDeduct= None
        self.len_beam=[]
        self.__ds= []
        self.__mdd= self.__member_details.copy()
        self.__ndd= self.__nodes_details.copy()
        self._bcd= self.__boundarycondition.copy()

    def __nodes_arrangement_for_members(self):                
        total_members= self.tm
        mem_nodes_cord = pd.DataFrame()
        type_m= []
        for kx in range(total_members):  
            mem_nodes = pd.DataFrame()   
            for jx in range(2):
                en = self.__member_details.iloc[kx,jx]   #getting the nodes 1 and 2 for a members in self.__member_details
                mem_nodes= pd.concat([mem_nodes,self.__nodes_details.loc[[en]]])
                mem_nodes_cord = pd.concat([mem_nodes_cord,self.__nodes_details.loc[[en]]])          # Getting Cordinates of members from self.__nodes_details
            if mem_nodes.iloc[0,1] == mem_nodes.iloc[1,1]:
                tym= 'Beam' 
            else:
                tym= 'Col'
            type_m.append(tym)
        self.__member_details['Type']= type_m    
        self.__cords_member_order= mem_nodes_cord


#------------------------------------------------------------------------------
    def __arrange_beam_column_nodes(self): 
        beams_mat= self.__member_details.loc[self.__member_details['Type']=='Beam']

        col_mat= self.__member_details.loc[self.__member_details['Type']=='Col']

        col_mat = col_mat.filter(['Node1', 'Node2'])
        beams_mat = beams_mat.filter(['Node1', 'Node2'])     
    
        p_y= self.__nodes_details.sort_values(by=['y','x', 'z'])
        p_y1= self.__nodes_details.sort_values(by=['y','x', 'z'])
        yy = p_y.y.unique()
        len_yy= len(yy)
    
        min_xyz= p_y.y.min()
        base= p_y[p_y.y.isin([min_xyz])] 
        self.baseN= base
        p_y= p_y.drop(base.index)
        column_detail= pd.DataFrame(((base.index.values).reshape((-1,1))),columns= ["Node1"])
        column_detail[["N1x","N1y","N1z"]]= base.to_numpy()   
    
        for i in range (1,len_yy):
                    
                min_xyz= p_y.y.min()   # p_y to p_y.y
                base1= p_y[p_y.y.isin([min_xyz])]
                base_index1= base1.index 
                column_detail[[f'Node{i+1}', f'N{i+1}x',f'N{i+1}y',f'N{i+1}z' ]]= ' '
                for j in range (len(base1)):
                    for k in range (len(base)):
                        if base1.iloc[j,0]== base.iloc[k,0]:
                            if base1.iloc[j,2]== base.iloc[k,2]:
                                node_new= base_index1[j]  
                                column_detail.iloc[k,i*4]= node_new
                                single_cord= (base1.loc[node_new,:])
                                column_detail.iloc[k,(i*4)+1]= single_cord['x']
                                column_detail.iloc[k,(i*4)+2]= single_cord['y']
                                column_detail.iloc[k,(i*4)+3]= single_cord['z']
                            
                p_y= p_y.drop(base_index1)
        column_detail.replace(' ', 0.0001, inplace=True)
    
        col_nodes= column_detail.filter([col for col in column_detail.columns if 'Node' in col])            
    
        column_detail[[f'C{i+1}' for i in range(len_yy-1) ]]= 0.0001
    
    
        for i in range (len(base)): 
            col_num= ((col_nodes.loc[i]).to_numpy()).flatten()
            col_num= np.delete(col_num, np.where(col_num == 0.0001))
            sets= list(itr.combinations(col_num, 2))
            len_sets= len(sets)
        
            ho=len_yy*4
            for j in range (len_sets):
                check=0
                for k in range (len(col_mat)):
                    if (sets[j][0]== col_mat.iloc[k,0]  or sets[j][0]== col_mat.iloc[k,1]):
                        if (sets[j][1]== col_mat.iloc[k,0]  or sets[j][1]== col_mat.iloc[k,1]):
                            column_detail.iloc[i,ho]= col_mat.index[k]
                            check= 1
                if check==1:
                    ho=ho+1

        L= np.sqrt(np.square(self.__nodes_details.loc[col_mat["Node1"]].values- self.__nodes_details.loc[col_mat["Node2"]].values).sum(axis=1)) 

        col_mat.insert(2, 'L', L)
        col_mat.insert(2, 'Story', ' ')
        for i in range(len_yy-1):
            col_mat.loc[col_mat.index.isin(column_detail[f'C{i+1}']), 'Story']= i


        #COLUMN_DETAIL CONTAINS ALL NODES AND THEIR CONTINUOUS COLUMN 
        #col_mat CONTAINS general details of each COLUMN 
#------------------------------------------------------------------------------
        # Beam Arrangement x-direction
        b_y= self.__nodes_details.sort_values(by=['y','x', 'z'])
    
        xx = b_y.x.unique()
        zz = b_y.z.unique()
        yyb= b_y.y.unique()
        beam_detail_x= pd.DataFrame()
        index_beam_mat= beams_mat.index
    
        d3= 1
        if len(yyb)== 1:     # Deals with only single beam model
            d3= 0
        
        for i in range (d3,len(yyb)):     #loop sorting beam story wise
            beam_1= b_y[b_y.y.isin([yyb[i]])]
            beam_1= beam_1.sort_values(by=['z', 'x'])
        
            for v in range (len(zz)):               #loop sorting beam along different z direction
                beam_11x= beam_1[beam_1.z.isin([zz[v]])]
            
                beam_index= beam_11x.index
                beam_index_df= pd.DataFrame(beam_index)
                beam_index_df.rename(columns = {'Index ':'x'}, inplace = True)
                beam_sets=list(itr.combinations(beam_index, 2))
            
                for j in range (len(beam_sets)):
                
                    for k in range (len(beams_mat)):
                        if (beam_sets[j][0]== beams_mat.iloc[k,0]  or beam_sets[j][0]== beams_mat.iloc[k,1]):
                            if (beam_sets[j][1]== beams_mat.iloc[k,0]  or beam_sets[j][1]== beams_mat.iloc[k,1]):
                            
                                if (len(beam_11x)>2):
                                
                                    xx_1st_Last_removed= (beam_index_df.tail(-1)).head(-1)
                                    n1_find= xx_1st_Last_removed.isin([beam_sets[j][0]])
                                    n1_con = n1_find.any()
                                    n2_find= xx_1st_Last_removed.isin([beam_sets[j][1]])
                                    n2_con = n2_find.any()                                
                                
                                else:
                                    n1_con = False
                                    n2_con = False
                            
                            
                            
                                new_beam= pd.DataFrame.from_dict([{'Beam' : index_beam_mat[k], 'Node1': beam_sets[j][0], 'Node2': beam_sets[j][1], 'Story': i, 'x': 0, 'z': zz[v],
                                       'Continous_at_Node1': n1_con, 'Continous_at_Node2': n2_con}])

                                beam_detail_x = pd.concat([beam_detail_x, new_beam], ignore_index = True)

        if  len (beam_detail_x.index) > 0:   
            beam_detail_x[['L']]=' '
            beam_detail_x.index= beam_detail_x.Beam

        for i in beam_detail_x.index:   
            pos1= beam_detail_x.loc[i,['Node1']].values[0]
            pos2= beam_detail_x.loc[i,['Node2']].values[0]
            beam_detail_x.loc[i,['L']]= np.sqrt(np.square((self.__nodes_details.loc[pos1]- self.__nodes_details.loc[pos2])).sum())
        

#------------------------------------------------------------------------------
        # Beam Arrangement in z direction                     
        beam_detail_z= pd.DataFrame()
        for i in range (d3,len(yyb)):
            beam_1z= b_y[b_y.y.isin([yyb[i]])]
            beam_1z= beam_1z.sort_values(by=['x', 'z'])
        
            for v in range (len(xx)):
                beam_11z= beam_1z[beam_1z.x.isin([xx[v]])]

                beam_indexz= beam_11z.index
                beam_index_df= pd.DataFrame(beam_indexz)
                beam_index_df.rename(columns = {'Index ':'z'}, inplace = True)
                beam_sets=list(itr.combinations(beam_indexz, 2))
            
                for j in range (len(beam_sets)):
                
                    for k in range (len(beams_mat)):
                        if (beam_sets[j][0]== beams_mat.iloc[k,0]  or beam_sets[j][0]== beams_mat.iloc[k,1]):
                            if (beam_sets[j][1]== beams_mat.iloc[k,0]  or beam_sets[j][1]== beams_mat.iloc[k,1]):
                            
                                if (len(beam_11z)>2):
                                
                                    xx_1st_Last_removed= (beam_index_df.tail(-1)).head(-1)
                                    n1_find= xx_1st_Last_removed.isin([beam_sets[j][0]])
                                    n1_con = n1_find.any()
                                    n2_find= xx_1st_Last_removed.isin([beam_sets[j][1]])
                                    n2_con = n2_find.any()           
                                else:
                                    n1_con = False
                                    n2_con = False
                            
                                new_beam= pd.DataFrame.from_dict([{'Beam' : index_beam_mat[k], 'Node1': beam_sets[j][0], 'Node2': beam_sets[j][1], 'Story': i, 'x': xx[v],'z': 0,
                                       'Continous_at_Node1': n1_con, 'Continous_at_Node2': n2_con, }])
                            
                                beam_detail_z = pd.concat([beam_detail_z, new_beam], ignore_index = True)
        
        if  len (beam_detail_z.index) > 0:     
            beam_detail_z[['L']]=' '
            beam_detail_z.index= beam_detail_z.Beam


        for i in beam_detail_z.index:   
            pos1= beam_detail_z.loc[i,['Node1']].values[0]
            pos2= beam_detail_z.loc[i,['Node2']].values[0]
            beam_detail_z.loc[i,['L']]= np.sqrt(np.square((self.__nodes_details.loc[pos1]- self.__nodes_details.loc[pos2])).sum())
                          
#------------------------------------------------------------------------------
    # Nodes Details
        node_details= pd.DataFrame()
        node_details[['Node']]= self.__nodes_details.index.values.reshape((-1,1))
        node_details[['Floor']]= ' '
        node_details[['Height']]= ' '
        yy = p_y1.y.unique() 
        bbb=0

        for i in yy:
            for k in node_details.index:
                nd= node_details.Node[k]
                for j in (p_y1.index):
                    if nd == j:
                        if p_y1.y[j]==i:
                            node_details.loc[k,'Floor'] = bbb
                            node_details.loc[k,'Height']= p_y1.y[j]     

            bbb=bbb+1
    
        node_details[['Beam1','Beam2','Beam3','Beam4','Col1','Col2']]= ' '
    
        for i in range (len(node_details)):
            node= node_details.loc[i,'Node']
            beam_numbers= beams_mat.index[(beams_mat['Node1']==node) | (beams_mat['Node2']==node)]

            column_numbers= col_mat.index[(col_mat['Node1']==node) | (col_mat['Node2']==node)]

            b_len= len(beam_numbers)
            c_len= len(column_numbers)
            
            for j in range (b_len):
                node_details.iloc[i,j+3]= beam_numbers[j]
        
            for k in range (c_len):
                node_details.iloc[i,k+7]= column_numbers[k]                
        
        node_details.replace(' ', 0 ,inplace=True)
        
        beam_details= pd.concat([beam_detail_x, beam_detail_z]) #, ignore_index=True
        
        self.beams_detail=beam_details.copy()     
        self.columns_detail= col_mat.copy()
        self.nodes_detail= node_details.copy()

        node_details.set_index("Node", inplace= True)

        beam_details[["WDedn1"]]= " "
        beam_details[["WDedn2"]]= " "

        for i in beam_details.index:
            n1= beam_details.at[i,"Node1"]
            n2= beam_details.at[i,"Node2"]
            c1= node_details.at[n1,"Col1"]
            c2= node_details.at[n1,"Col2"]

            c3= node_details.at[n2,"Col1"]
            c4= node_details.at[n2,"Col2"]

            if c1==0:
                c1_d= 10
                c1_b= 10
            else:
                c1_d= self.__member_details.at[c1,'d']
                c1_b= self.__member_details.at[c1,'b']
            
            if c2==0:
                c2_d= 10
                c2_b= 10
            else:
                c2_d= self.__member_details.at[c2,'d']
                c2_b= self.__member_details.at[c2,'b']

            if c3==0:
                c3_d= 10
                c3_b= 10
            else:
                c3_d= self.__member_details.at[c3,'d']
                c3_b= self.__member_details.at[c3,'b']

            if c4==0:
                c4_d= 10
                c4_b= 10
            else:
                c4_d= self.__member_details.at[c4,'d']
                c4_b= self.__member_details.at[c4,'b']

            if (beam_detail_x.index.isin([i]).any()== True ):
                beam_details.at[i,"WDedn1"]= np.where((c1_d >= c2_d), c1_d, c2_d)

                beam_details.at[i,"WDedn2"]= np.where((c3_d >= c4_d), c3_d, c4_d)

            if (beam_detail_z.index.isin([i]).any()== True ):
                beam_details.at[i,"WDedn1"]= np.where((c1_b >= c2_b), c1_b, c2_b)

                beam_details.at[i,"WDedn2"]= np.where((c3_b >= c4_b), c3_b, c4_b)

        self.__bd_LDeduct= beam_details.filter(['L', 'WDedn1', 'WDedn2'])
        
        self.__bd_LDeduct["L_clear"]= self.__bd_LDeduct["L"]-((self.__bd_LDeduct['WDedn1']/2000)+(self.__bd_LDeduct['WDedn2']/2000))

        self.beam_details_with_deduction= self.beams_detail.copy()
        self.__beams_detail_preP= self.beams_detail.copy()     
        self.__columns_detail_preP= self.columns_detail.copy()
        self.__nodes_detail_preP= self.nodes_detail.reset_index().copy()


    def __arrange_all(self):

        beams_mat= self.__member_details.loc[self.__member_details['Type']=='Beam']
        beams_mat = beams_mat.filter(['Node1', 'Node2']) 

        fun_id1 = arrange_col_Frame.remote(self.__member_details, self.__nodes_details)
        fun_id2 = arrange_beam_FrameX.remote(beams_mat, self.__nodes_details)
        fun_id3 = arrange_beam_FrameZ.remote(beams_mat, self.__nodes_details)

        gf1, self.gf2, self.gf3 = ray.get([fun_id1, fun_id2, fun_id3])

        baseN= gf1[0]
        col_mat=    gf1[1]
        columns_detail= gf1[2]


        beam_detail_x= self.gf2
        beam_detail_z= self.gf3
        

        bd_LDeduct, real_beam_details, beam_details, nodesD, node_details = ray.get(arrange_nodes_Frame.remote(self.__member_details, self.__nodes_details,beams_mat, beam_detail_x, beam_detail_z, col_mat ))


        self.baseN= baseN
        self.__bd_LDeduct= bd_LDeduct.copy()

        self.beams_detail=real_beam_details.copy()     
        self.columns_detail= columns_detail.copy()
        self.nodes_detail= nodesD.copy()

        self.beam_details_with_deduction= self.beams_detail.copy()
        self.__beams_detail_preP= beam_details.copy()     
        self.__columns_detail_preP= self.columns_detail.copy()
        self.__nodes_detail_preP= node_details.reset_index().copy()


    def __autoflooring(self):  

        stories= len(self.__nodes_details.y.unique())  
        x_uni= self.__nodes_details.x.unique()
        z_uni= self.__nodes_details.z.unique()
        total_floors_pd= pd.DataFrame()
        slab_no= 1
        self.__slab_details = pd.DataFrame()

        for i in range(1,stories):
            nodes_floor= self.nodes_detail[self.nodes_detail["Floor"]==i]
            nodes_floor_detail= self.__nodes_details.loc[nodes_floor.Node]

            floors_pd= pd.DataFrame(columns=x_uni, index=z_uni) 

            for j in z_uni:
                for k in x_uni:
                    node_no= nodes_floor_detail.where((nodes_floor_detail.x == k) & (nodes_floor_detail.z == j))
                    if node_no.isnull().values.all(axis=0).all() == True:
                        floors_pd.at[j,k]=0 
                    else:
                        node_no.dropna(inplace=True)
                        floors_pd.at[j,k]=node_no.index.item()  
            total_floors_pd=pd.concat([total_floors_pd,floors_pd])    

            for l in range (len(z_uni)-1):
                for m in range (len(x_uni)-1): 
                    N1= floors_pd.iat[l,m]
                    N2= floors_pd.iat[l,m+1]
                    N3= floors_pd.iat[l+1,m]
                    N4= floors_pd.iat[l+1,m+1]
                    if N1==0 or N2==0 or N3==0 or N4==0:
                        continue
                    else:
                        slab_details= pd.DataFrame({"Node1": N1, "Node2": N2, "Node3": N3, "Node4": N4,	"Thickness(mm)": 150, "FF(kN/m2)": -1, "LL(kN/m2)": -3, "Waterproofing(kN/m2)":0, "Floor": i }, index=[slab_no]) 
                        slab_no= slab_no+1
                    
                    self.__slab_details= pd.concat([self.__slab_details,slab_details])  


# #------------------------------------------------------------------------------
    def __slab_beam_load_transfer(self):
        s1= self.__slab_details.iloc[:,0:4]
        bd= self.beams_detail.copy()
        slab_pd= pd.DataFrame()


        for i in (s1.index):
            beam_no= []
            length = []
            
            for j in (bd.index):
                
                yes= bd.loc[j,["Node1","Node2"]].isin(s1.loc[i])
                if yes.all():
                    beam_no.append(j) 
                    length.append(bd.loc[j,["L"]].values[0])

                index_new= [
                np.array([i, i, i, i])
                #np.array(["Beam1", "Center", "Node2"]),
                ] 

            
            if length[0]== length[2] and length[0]== length[1] and length[2]== length[3] and length[1]== length[3]:                     # SQUARE SLAB
                t_l_dist= ["Triangular","Triangular","Triangular","Triangular"]
                type_l= [1,1,1,1] 
                h_t = length[0]/2               # Heigth of triangle
                h= [h_t,h_t,h_t,h_t]            # Heigth of triangle for 4 triangles
                
                # l1 , l2, l3 are the zone of loading shape with range
                l1= [0,0]                   # l1 TRAINGULAR part of trapezoidal loading LOAD 
                l2 = [0,0]                   # l2 rectangle part of trapezoidal loading LOAD
                l3 = [0,0]                  # l3 TRAINGULAR part of trapezoidal loading LOAD  
                l4=  [0,length[0]/2,length[0]]   # l4 traingular part of  traingular distribution in 2 way slab 
                l5=  [0,0]                       # l5 rectangle  part of 1 way slab



            if length[0]!= length[2] and length[0]== length[1] and length[2]== length[3] and length[1]!= length[3]:                     # Rectangular SLAB
                
                if (max(length[0],length[2])/min(length[0],length[2]))<=2:      # Checking 2 way slab for distribution

                    if length[0] > length[2]:                       
                        h_t=length[2] /2                     # Heigth of triangle and trapezoidal
                        h= [h_t,h_t,h_t,h_t]

                        t_l_dist= ["Trapezoidal","Trapezoidal","Triangular","Triangular"] 
                        type_l = [2,2,1,1]

                        # l1 , l2, l3 are the zone of loading shape with range
                        l1= [0,length[2]/2]                               # l1 TRAINGULAR part of trapezoidal loading LOAD            
                        l2 = [length[2]/2,(length[0]-(length[2]/2))]       # l2 rectangle part of trapezoidal loading LOAD
                        l3 = [(length[0]-(length[2]/2)),length[0]]         # l3 TRAINGULAR part of trapezoidal loading LOAD  
                        l4=  [0,length[2]/2,length[2]]          # l4 traingular part of traingular in 2 way slab 
                        l5=  [0,0]                                     # l5 rectangle  part of 1 way slab


 

                    if length[2] >= length[0]:               
                        h_t=length[0] /2                    # Heigth of triangle and trapezoidal
                        h= [h_t,h_t,h_t,h_t]

                        t_l_dist= ["Triangular","Triangular","Trapezoidal","Trapezoidal"] 
                        type_l = [1,1,2,2]

                        # l1 , l2, l3 are the zone of loading shape with range
                        l1= [0,length[0]/2]                                  # l1 TRAINGULAR part of trapezoidal loading LOAD            
                        l2 = [length[0]/2,(length[2]-(length[0]/2))]         # l2 rectangle part of trapezoidal loading LOAD
                        l3 = [(length[2]-(length[0]/2)),length[2]]           # l3 TRAINGULAR part of trapezoidal loading LOAD  
                        l4=  [0,length[0]/2,length[0]]                         # l4 traingular part of small traingular in 2 way slab 
                        l5=  [0,0]                                     # l5 rectangle  part of 1 way slab                    

                else:                                           # Below code for 1 way slab distribution
                    if length[0] > length[2]:
                        h_t=length[2] /2
                        h= [h_t,h_t,0,0]
                        l1= [0,0]
                        l2= [0,0]
                        l3= [0,0]
                        l4= [0,0]
                        l5= [0,length[0]]
                        t_l_dist= ["Rectangle","Rectangle","None","None"] 
                        type_l = [3,3,0,0] 

                    if length[2] > length[0]:
                        h_t=length[0] /2
                        h= [0,0,h_t,h_t]
                        l1= [0,0]
                        l2= [0,0]
                        l3= [0,0]
                        l4= [0,0]
                        l5= [0,length[2]]                    
                        t_l_dist= ["None","None","Rectangle","Rectangle"] 
                        type_l = [0,0,3,3] 

            ld1=[l1,l1,l1,l1]
            ld2=[l2,l2,l2,l2]
            ld3=[l3,l3,l3,l3]
            ld4= [l4,l4,l4,l4]
            ld5= [l5,l5,l5,l5]
            t= self.__slab_details.loc[i,['Thickness(mm)']].values[0]
            ff= self.__slab_details.loc[i,['FF(kN/m2)']].values[0]
            ll= self.__slab_details.loc[i,['LL(kN/m2)']].values[0]
            wp= self.__slab_details.loc[i,['Waterproofing(kN/m2)']].values[0]
            slab_t= [t,t,t,t]
            ff1=[ff,ff,ff,ff]
            wp1= [wp,wp,wp,wp] 
            ll1= [ll,ll,ll,ll]
            sb_s= pd.DataFrame({'Beam': beam_no, 'L':length,  'Distribution_type': t_l_dist, 'Type_loading': type_l, "Height_l": h, 'LD1':ld1, 'LD2':ld2, 'LD3':ld3, 'LD4':ld4,'LD5':ld5,'slab_t': slab_t, 'FF': ff1, 'WP': wp1, 'LL': ll1   },index=index_new) 
            slab_pd= pd.concat([slab_pd, sb_s])
            
        self.slab_pd= slab_pd



    def __properties(self):
   
        rec_crossSection= self.__member_details.index[self.__member_details['b']>0].tolist()
        cir_crossSection= self.__member_details.index[self.__member_details['b']==0].tolist()

        d=self.__member_details.loc[rec_crossSection,"d"]/2
        b=self.__member_details.loc[rec_crossSection,"b"]/2

        
        self.rec_crossSection = rec_crossSection
        

        if len(rec_crossSection) > 0:
            cross_section = "Rectangular"
            self.__member_details.loc[rec_crossSection,"Area"]=(self.__member_details.loc[rec_crossSection,"b"]*self.__member_details.loc[rec_crossSection,"d"])/(10**6)  #in m2 #8

            self.__member_details.loc[rec_crossSection,"Iz" ]= (self.__member_details.loc[rec_crossSection,"b"]*(self.__member_details.loc[rec_crossSection,"d"]**3)/12)/(10**12)                # M.O.I in x direction of local coordinate in m4
            self.__member_details.loc[rec_crossSection,"Iy"]= (self.__member_details.loc[rec_crossSection,"d"]*(self.__member_details.loc[rec_crossSection,"b"]**3)/12)/(10**12)      #in m4
            self.__member_details.loc[rec_crossSection,"J"]= ((d*(b**3))*( (16/3) -(3.36*(b/d)*(1-((b**4)/(12*(d**4)))))))/(10**12)     #in m4
        
        if len(cir_crossSection) > 0:
            cross_section = "circular"

            self.__member_details.loc[cir_crossSection,"Area"]= (np.pi*self.__member_details.loc[rec_crossSection,"d"]*self.__member_details.loc[rec_crossSection,"d"]/4)/(10**6)  #in m2 #8

            self.__member_details.loc[cir_crossSection,"Iz" ]= ((np.pi()*self.__member_details.loc[rec_crossSection,"d"]**4)/64)/(10**12)                # M.O.I in x direction of local coordinate in m4

            self.__member_details.loc[cir_crossSection,"Iy"]= ((np.pi()*self.__member_details.loc[rec_crossSection,"d"]**4)/64)/(10**12)      #in m4

            self.__member_details.loc[cir_crossSection,"J"]= self.__member_details["Iz"]+  self.__member_details["Iy"]   #in m4



        for i in range (len(self.__Mproperties.index)):

            if self.__Mproperties.iat[i,0]=="All":
                E = self.__Mproperties.iat[i,4]*1000              #in N and m2  
                mu= self.__Mproperties.iat[i,5]                    #poisson's ratio
                self.__concrete_density_All= self.__Mproperties.iat[i,3]
                density= self.__concrete_density_All            #24.02615            # kN/m3
                alpha = self.__Mproperties.iat[i,6]       #thermal coefficient
                critdamp= self.__Mproperties.iat[i,7]
                G = self.__Mproperties.iat[i,8]*1000    
                self.__concrete_density_beam= density
                self.__concrete_density_col= density
                self.__concrete_densitySlab= density

                self.__member_details['Modulus of Elasticity']= E
                self.__member_details['Poisson Ratio']= mu
                self.__member_details['Density']= density
                self.__member_details['Thermal Coefficient']= alpha
                self.__member_details['Critical Damping']= critdamp
                self.__member_details['Modulus of Rigidity']= G

            if self.__Mproperties.iat[i,0]=="Beam":
                beam_index= self.__member_details[self.__member_details['Type']=="Beam"].index
                E = self.__Mproperties.iat[i,4]*1000              #in N and m2  
                mu= self.__Mproperties.iat[i,5]                    #poisson's ratio
                density= self.__Mproperties.iat[i,3]            #24.02615            # kN/m3
                alpha = self.__Mproperties.iat[i,6]       #thermal coefficient
                critdamp= self.__Mproperties.iat[i,7]
                G = self.__Mproperties.iat[i,8]*1000       
                self.__concrete_density_beam= density

                self.__member_details.loc[beam_index,'Modulus of Elasticity']= E
                self.__member_details.loc[beam_index,'Poisson Ratio']= mu
                self.__member_details.loc[beam_index,'Density']= density
                self.__member_details.loc[beam_index,'Thermal Coefficient']= alpha
                self.__member_details.loc[beam_index,'Critical Damping']= critdamp
                self.__member_details.loc[beam_index,'Modulus of Rigidity']= G

            if self.__Mproperties.iat[i,0]=="Column":
                Col_index= self.__member_details[self.__member_details['Type']=="Col"].index
                E = self.__Mproperties.iat[i,4]*1000              #in N and m2  
                mu= self.__Mproperties.iat[i,5]                    #poisson's ratio
                density= self.__Mproperties.iat[i,3]            #24.02615            # kN/m3
                alpha = self.__Mproperties.iat[i,6]       #thermal coefficient
                critdamp= self.__Mproperties.iat[i,7]
                G = self.__Mproperties.iat[i,8]*1000       
                self.__concrete_density_col= density

                self.__member_details.loc[Col_index,'Modulus of Elasticity']= E
                self.__member_details.loc[Col_index,'Poisson Ratio']= mu
                self.__member_details.loc[Col_index,'Density']= density
                self.__member_details.loc[Col_index,'Thermal Coefficient']= alpha
                self.__member_details.loc[Col_index,'Critical Damping']= critdamp
                self.__member_details.loc[Col_index,'Modulus of Rigidity']= G

            if self.__Mproperties.iat[i,0]=="Slab":
                self.__concrete_densitySlab= 25     #self.__Mproperties.iat[i,3]

        if self.__self_weight== False:
            self.__concrete_density = 0
            self.__concrete_density_beam= 0
            self.__concrete_density_col= 0
            self.__concrete_densitySlab = 0
            
    def __member_detailing(self):
        self.beams_detail= self.__beams_detail_preP.copy()     
        self.columns_detail= self.__columns_detail_preP.copy()
        self.nodes_detail= self.__nodes_detail_preP.copy()


        self.columns_detail.loc[self.columns_detail.index,['b','d','Area', 'Iz','Iy','J']]=self.__member_details.loc[self.columns_detail.index,['b','d','Area', 'Iz','Iy','J']]

        self.beams_detail.loc[self.beams_detail.index,['b','d','Area', 'Iz','Iy','J']]=self.__member_details.loc[self.beams_detail.index,['b','d','Area', 'Iz','Iy','J']]

        # Stiffness Calculation of Members
        self.beams_detail[['Stiffness_Factor']]=((self.beams_detail.Iz)/(self.beams_detail.L)).values.reshape((-1,1))

        self.columns_detail[['Stiffness_Factor']]= ' '

        for i in self.columns_detail.index:
            max_I= max(self.columns_detail.at[i,'Iz'],self.columns_detail.at[i,'Iy'])
            Length= self.columns_detail.at[i,'L']
            self.columns_detail.at[i,'Stiffness_Factor']=max_I/Length

#------------------------------------------------------------------------------
    # Stiffness Calculation at Nodes  
        self.nodes_detail[['Stiffness']]= ' '
        p_y=self.__nodes_details.sort_values(by=['y','x', 'z'])
        building_base= p_y[p_y.y.isin([p_y.y.min()])].index
        building_base=building_base.values.reshape((1,len(building_base)))

        for i in range(len(self.nodes_detail)):
            
            b1=self.nodes_detail.at[i,'Beam1']
            b2=self.nodes_detail.at[i,'Beam2']
            b3=self.nodes_detail.at[i,'Beam3']
            b4=self.nodes_detail.at[i,'Beam4']
            Kb=0
            Kc=0
            
            bottom_node=self.nodes_detail.Node[i]==building_base
            
            if bottom_node.any():
                Kb= np.inf  
            else: 
                for j in range (1,5):
                    ff= locals()["b"+str(j)]
                    if ff!=0:
                        Kb = Kb + self.beams_detail.Stiffness_Factor[ff]
        
            c1= self.nodes_detail.Col1[i]
            c2= self.nodes_detail.Col2[i]  

            for j in range (1,3):
                ff= locals()["c"+str(j)]
                if ff!=0:
                    Kc = Kc + self.columns_detail.Stiffness_Factor[ff]        
            K= Kb+Kc
            self.nodes_detail.loc[i,'Stiffness']= K

            
        self.nodes_detail= self.nodes_detail.set_index('Node')

    def __swayORnot(self):
        self.def_story= []
        self.story_axial_load= []

        self.stability_index= []
        self.sway=[]
        for i in range (len(self.__ASD.index)):
            avg_def= 0
            if (abs(self.__ASD.at[i,"Avg. ux (mm)"])) >= (abs(self.__ASD.at[i,"Avg. uz (mm)"])):
                avg_def= abs(self.__ASD.at[i,"Avg. ux (mm)"]) 
                    
            else:
                avg_def= abs(self.__ASD.at[i,"Avg. uz (mm)"])
                
                    
            self.def_story.append(avg_def)

            cols= self.columns_detail[self.columns_detail.Story.isin([i])].index.to_list()
            
            SaxialF= self.axial_forces_pd.loc[cols].sum().item()   # in kN

            self.story_axial_load.append(SaxialF) 
            
            if i!=0:
                stability_index= (SaxialF*avg_def)/(self.__SeismicShear.iat[i,0] * (self.__ASD.iat[i,3]*1000))
            else:
                stability_index= 0
            
            if (self.__SeismicShear.iat[i,0] == 0):
                stability_index= 0

            self.stability_index.append(stability_index)
            
            if stability_index > self.__col_stablity_index:
                self.sway.append(1)
            else:
                self.sway.append(0)

    def __effectiveLength(self):
        # Sway or non-sway columns

        if self.__seismic_def_status == True:
            self.__swayORnot()

        # Effective Length Calculation of columns
        self.columns_detail[['Beta1']]= ' '
        self.columns_detail[['Beta2']]= ' '
        self.columns_detail[['Lef_coef']]= ' '
        self.columns_detail[['Lef']]= ' '
        self.columns_detail[['Type']]= ' '

        for i in (self.columns_detail.index):
            
            story= self.columns_detail.at[i,'Story']
            
            if self.__seismic_def_status == True:
                sway= self.sway[story] 
            else:
                sway = 0

            Kc= self.columns_detail.Stiffness_Factor[i]
            lj= self.columns_detail.Node1[i]
            uj= self.columns_detail.Node2[i]
            lower_joint= self.nodes_detail.Stiffness[lj]
            upper_joint= self.nodes_detail.Stiffness[uj]
            B1= Kc/upper_joint
            B2= Kc/lower_joint
            self.columns_detail.loc[i,'Beta1']= B1
            self.columns_detail.loc[i,'Beta2']= B2
            if sway==0:
                K= (1+(0.145*(B1+B2))-(0.265*B1*B2))/ (2-(0.364*(B1+B2))+(0.247*B1*B2))
            if sway==1:
                K= np.sqrt((1+(0.2*(B1+B2))-(0.12*B1*B2))/ (1-(0.8*(B1+B2))+(0.6*B1*B2)))
                if K<1.2:
                    K=1.2

            self.columns_detail.loc[i,'Lef_coef']= K
            self.columns_detail.loc[i,'Lef']= K*self.columns_detail.L[i]
            s_ratio= self.columns_detail.Lef[i] /((max(self.columns_detail.b[i],self.columns_detail.d[i]))/1000)
            if s_ratio>12:
                self.columns_detail.loc[i,'Type']= "Long"
            elif s_ratio<3:
                self.columns_detail.loc[i,'Type']= "Pedestal"
            else:
                self.columns_detail.loc[i,'Type']= "Short"


    def __floorLoading(self):
        s1= self.__slab_details.iloc[:,0:4]
        bd= self.beams_detail.copy()
        floor_loads= pd.DataFrame()
        
        for i in (s1.index):
            beam_no= []
            length = []
            for j in (bd.index):
                yes= bd.loc[j,["Node1","Node2"]].isin(s1.loc[i])
                if yes.all():
                    beam_no.append(j) 
                    length.append(bd.loc[j,["L"]].values[0])


            t= self.__slab_details.loc[i,['Thickness(mm)']].values[0]
            ff= self.__slab_details.loc[i,['FF(kN/m2)']].values[0]
            ll= self.__slab_details.loc[i,['LL(kN/m2)']].values[0]
            wp= self.__slab_details.loc[i,['Waterproofing(kN/m2)']].values[0]

            #floor loads
            sw= -1*self.__concrete_densitySlab*t*length[0]*length[2]/1000      #kN
            odl= (ff+wp)*length[0]*length[2]        #kN
            oll= ll*length[0]*length[2]
            story= bd.loc[beam_no[0],'Story']
            floads_pd= pd.DataFrame({'Self-Weight(kN)': sw, "Other-Dead-Loads":odl,  'LL(kN)': oll, 'Story':story},index=[i])
            floor_loads= pd.concat([floor_loads, floads_pd])
        self.__floor_loads= floor_loads
        

    def __tloads(self):
        #updating_member_details= self.properties()
        if self.__slabload_there==1:
            self.__slabvdl()


        Col_selfweight=[]
        Beam_selfweight=[]        
        col_odl= []
        col_story= []
        beam_odl= []
        beam_story= []

        col_index= []
        beam_index= []

        for i in (self.__member_details.index):

            if (self.__member_details.loc[i,"Type"]=='Col'):
                sw= self.__concrete_density_col*((self.__member_details.loc[i,'b']*self.__member_details.loc[i,'d']/1000000))
                odl= self.__member_details.loc[i,'xUDL']          #other deal loads (xudl beacuse it converts into yudl)

                self.__member_details.loc[i,'xUDL']= self.load_combo.iloc[0,0]*(odl+(-sw))    #lself.load_combo.Dead_Load.values[0] 
                #Converting selfwight for GLOBAL Y AXIS for members along y-Axis

                sw1= -1*sw*self.columns_detail.loc[i,'L']
                odl1= odl*self.columns_detail.loc[i,'L']

                Col_selfweight.append(sw1)
                col_odl.append(odl1)
                col_story.append(self.columns_detail.loc[i,'Story'])
                col_index.append(i)

            else: 
                sw= self.__concrete_density_beam*((self.__member_details.loc[i,'b']*self.__member_details.loc[i,'d']/1000000))
                odl= self.__member_details.loc[i,'yUDL']

                self.__member_details.loc[i,'yUDL']= self.load_combo.iloc[0,0]*(odl+(-sw)) 

                sw1= -1*sw*self.beams_detail.loc[i,'L']                        #self.__bd_LDeduct.loc[i,'L_clear']
                odl1= odl*self.beams_detail.loc[i,'L']                      #self.__bd_LDeduct.loc[i,'L_clear']

                Beam_selfweight.append(sw1)
                beam_odl.append(odl1)
                beam_story.append(self.beams_detail.loc[i,'Story'])
                beam_index.append(i)
                
        self.__column_loads= pd.DataFrame({"Self-Weight(kN)": Col_selfweight, "Other-Dead-Loads":col_odl, "Story": col_story},index=col_index)
        self.__beam_loads= pd.DataFrame({"Self-Weight(kN)": Beam_selfweight, "Other-Dead-Loads":beam_odl, "Story": beam_story},index=beam_index)   #other load only contains vertical loads in global axis
        self.__Storylumpload()
        
        
    def __Storylumpload(self):
        self.__column_Lumploads= pd.DataFrame()
        self.__beam_Lumploads= pd.DataFrame()

        stories= len(self.__nodes_details.y.unique())
        y_uni= self.__nodes_details.y.unique()
        y_uni= np.insert(y_uni,0,y_uni[0])
        Loadings_members= pd.DataFrame()
        Loadings_floor= pd.DataFrame()
        story_ht= []
        story_no= []
        for i in range(stories):
            ht= y_uni[i+1]- y_uni[0]
            story_ht.append(ht)
            story_no.append(i)

            beam_load_story= self.__beam_loads.loc[self.__beam_loads['Story']==i]

            if i == 0:               
                col_load_story= self.__column_loads.loc[self.__column_loads['Story']==i] / 2

            elif i == stories-1:
                col_load_story= self.__column_loads.loc[self.__column_loads['Story']==(i-1)] / 2

            else:
               col_load_story= pd.concat([(self.__column_loads.loc[self.__column_loads['Story']==(i-1)] / 2) , (self.__column_loads.loc[self.__column_loads['Story']==(i)] / 2)],axis=0)
               
               
            beam_load_sum= beam_load_story.sum().to_frame().T
            col_load_sum= col_load_story.sum().to_frame().T
            
            self.__column_Lumploads= pd.concat([self.__column_Lumploads,col_load_sum],axis=0)
            self.__beam_Lumploads= pd.concat([self.__beam_Lumploads,beam_load_sum],axis=0)
            
             
            if self.__slabload_there==1:
                floor_load_story= self.__floor_loads.loc[self.__floor_loads['Story']==i]
            
                #.to_frame().T converts series to DataFrame
                floor_sum= floor_load_story.sum().to_frame().T
                floor_sum.index=[i] 
                #.to_frame().T converts series to DataFrame

                Loadings_floor= pd.concat([Loadings_floor, floor_sum ])        
        
                self.__FL=Loadings_floor

        member_sw= self.__column_Lumploads['Self-Weight(kN)']+ self.__beam_Lumploads['Self-Weight(kN)']
        member_odl= self.__column_Lumploads['Other-Dead-Loads']+ self.__beam_Lumploads['Other-Dead-Loads']

        story_ht_pd= pd.DataFrame({"Ht":story_ht})
        self.__column_Lumploads.drop(['Story'], axis=1,inplace=True)
        self.__beam_Lumploads.drop(['Story'], axis=1,inplace=True)
        
        self.__column_Lumploads.index=story_no
        self.__beam_Lumploads.index=story_no

        Loadings_members= pd.concat([member_sw,member_odl],axis=1)
        Loadings_members.index=story_no


        if self.__slabload_there==1:
            Loadings_floor.drop(['Story'], axis=1,inplace=True)

            TSL_self_weight= Loadings_members.loc[:,'Self-Weight(kN)']+ Loadings_floor.loc[:,'Self-Weight(kN)']
            TSL_other_dead_load= Loadings_members.loc[:,'Other-Dead-Loads']+ Loadings_floor.loc[:,'Other-Dead-Loads']
            TSL_LL= Loadings_floor.loc[:,'LL(kN)']

            self.story_lumploads = pd.concat([TSL_self_weight,TSL_other_dead_load,TSL_LL, story_ht_pd],axis=1)
        else:
            TSL_self_weight= Loadings_members.loc[:,'Self-Weight(kN)']
            TSL_other_dead_load= Loadings_members.loc[:,'Other-Dead-Loads']

            self.story_lumploads = pd.concat([TSL_self_weight,TSL_other_dead_load,story_ht_pd],axis=1)

    def __slabvdl(self):
        dlvdl= pd.DataFrame()           # DEAD LOAD DataFrame for slab
        llvdl= pd.DataFrame()           # LIVE LOAD DataFrame for slab
        slabd= self.slab_pd.copy()

        slab_name= slabd.index.unique().tolist()
        RA= np.array([])
        RB= np.array([])
        MA= np.array([])
        MB= np.array([])

        RAl= np.array([])
        RBl= np.array([])
        MAl= np.array([])
        MBl= np.array([])

        for j in range(0,len(slab_name)):
            sb= slabd.loc[slab_name[j]]
            sb.index= [i for i in range (len(sb))]

            for i in range (0,len(sb)):
                
                L= int (sb.L[i])
                H = int(sb.Height_l[i])
                w= -(self.load_combo.iat[0,0]*((-self.__concrete_densitySlab*(sb.slab_t[i])/1000) + (sb.FF[i])+ (sb.WP[i]))*H)

                wl= -(self.load_combo.iat[0,1]*sb.LL[i]*H)

                if sb.Type_loading[i]==1:
                    Ra= w*L/4
                    Rb= Ra
                    Ma= 5*w*L*L/96
                    Mb = Ma

                    Ral= wl*L/4
                    Rbl= Ral
                    Mal= 5*wl*L*L/96
                    Mbl = Mal
                    
                if sb.Type_loading[i]==2:
                    Ra= w*(L-H)/2
                    Rb= Ra
                    Ma=( w*(5*L*L*((L*L)-(2*H*H)) + (5*L*H*H*H)))/(60*L*L)
                    Mb = Ma

                    Ral= wl*(L-H)/2
                    Rbl= Ral
                    Mal= ( wl*(5*L*L*((L*L)-(2*H*H)) + (5*L*H*H*H)))/(60*L*L)
                    Mbl = Mal

                if sb.Type_loading[i]==0:
                    Ra= 0  
                    Rb= 0
                    Ma= 0
                    Mb = 0

                    Ral= 0
                    Rbl= 0
                    Mal= 0
                    Mbl = 0

                if sb.Type_loading[i]==3:
                    Ra= w*(L*H)/2
                    Rb= Ra
                    Ma= w*H*(L*L)/12
                    Mb = Ma 

                    Ral= wl*(L*H)/2
                    Rbl= Ral
                    Mal= wl*H*(L*L)/12
                    Mbl = Mal                    

                RA= np.hstack((RA,Ra))
                RB= np.hstack((RB,Rb))
                MA= np.hstack((MA,Ma))
                MB= np.hstack((MB,Mb))

                RAl= np.hstack((RAl,Ral))
                RBl= np.hstack((RBl,Rbl))
                MAl= np.hstack((MAl,Mal))
                MBl= np.hstack((MBl,Mbl))
       
        dlvdl['Ra']= RA
        dlvdl['Rb']= RB
        dlvdl['Ma']= MA
        dlvdl['Mb']= MB

        llvdl['Ra']= RAl
        llvdl['Rb']= RBl
        llvdl['Ma']= MAl
        llvdl['Mb']= MBl

        dlvdl.index= slabd.Beam.to_list()
        llvdl.index= slabd.Beam.to_list()
        self.__dlvdl= dlvdl
        self.__llvdl= llvdl

    def __stiffnessbeam(self):   
         
        cords= self.__cords_member_order.to_numpy()
        nodes= self.__nodes_details.to_numpy()
        tn= self.tn                               #self.nodes()
        tm= self.tm                               #self.members()
        mem_load= self.__member_details.loc[:,'xUDL':'zUDL']*1000             # CHanging from kN to N
        
        slabload= self.__slabload_there

        Kg= np.zeros((tn*6,tn*6))
        global_nodal_forces= np.zeros(tn*6)
        
        KK_Local= np.empty((tm,12,12))
        
        m_L= (mem_load.to_numpy())     #self.load_combo.iat[0,0]*
        local_forces=np.array([])
        mmm=0
        ebc= np.array([[],[],[],[],[],[],[],[],[],[],[],[]])      # All the elments matrix in ebc     
        T_trans= np.empty((tm,12,12))
        eleno=int(0)    
        
        member_no= 0
        mem_index= self.member_list

        for ii in range(0,tm*2,2):
            
            mem_name= mem_index[member_no]

            position= int(ii/2) 

            A = self.__member_details.iat[position,8]
            Iz = self.__member_details.iat[position,9]
            Iy = self.__member_details.iat[position,10]
            J = self.__member_details.iat[position,11] 
            E = self.__member_details.iat[position,12]
            G = self.__member_details.iat[position,17]

            i=ii
        
            h=ii+1
        
            L= np.sqrt(np.sum((cords[h,:]-cords[i,:])**2))     # Length in m
            self.len_beam.append(L)

            k1 = E*A/L;
            k2 = 12*E*Iz/(L*L*L);
            k3 = 6*E*Iz/(L*L);
            k4 = 4*E*Iz/L;
            k5 = 2*E*Iz/L;
            k6 = 12*E*Iy/(L*L*L);
            k7 = 6*E*Iy/(L*L);
            k8 = 4*E*Iy/L;
            k9 = 2*E*Iy/L;
            k10 = G*J/L;
        

            k = np.array ([[k1, 0, 0, 0, 0, 0, -k1, 0, 0, 0, 0, 0],
                           [0, k2, 0, 0, 0, k3, 0, -k2, 0, 0, 0, k3],
                           [0, 0, k6, 0, -k7, 0, 0, 0, -k6, 0, -k7, 0],
                           [0, 0, 0, k10, 0, 0, 0, 0, 0, -k10, 0, 0],
                           [0, 0, -k7, 0, k8, 0, 0, 0, k7, 0, k9, 0],
                           [0, k3, 0, 0, 0, k4, 0, -k3, 0, 0, 0, k5],
                           [-k1, 0, 0, 0, 0, 0, k1, 0, 0, 0, 0, 0],
                           [0, -k2, 0, 0, 0, -k3, 0, k2, 0, 0, 0, -k3],
                           [0, 0, -k6, 0, k7, 0, 0, 0, k6, 0, k7, 0],
                           [0, 0, 0, -k10, 0, 0, 0, 0, 0, k10, 0, 0],
                           [0, 0, -k7, 0, k9, 0, 0, 0, k7, 0, k8, 0],
                           [0, k3, 0, 0, 0, k5, 0, -k3, 0, 0, 0, k4]], dtype="object")
        
                   
            #*** NEW TRANSFORMATION MATRIX***
            alpha= 0  # Rotation about logitudinal axis i.e rotation of cross section in degrees
            Cx=(cords[h,0]-cords[i,0])/L;
            Cy= (cords[h,1]-cords[i,1])/L;
            Cz= (cords[h,2]-cords[i,2])/L;

            Cxz= np.sqrt(Cx**2 + Cz**2)

            cg= np.cos(np.radians(alpha))              
            sg= np.sin(np.radians(alpha))              

            sb= Cy



            if (cords[i,0] == cords[h,0]  and cords[i,2] == cords[h,2]):
                    t= np.array ([[0,sb,0],
                           [-sb*cg,0,sg],
                           [sb*sg,0,cg]])
            else:
                    t= np.array ([[Cx,Cy,Cz],
                           [(-(Cx*Cy*cg)-(Cz*sg))/Cxz, Cxz*cg,(-(Cy*Cz*cg)+(Cx*sg))/Cxz],
                           [((Cx*Cy*sg)-(Cz*cg))/Cxz, -Cxz*sg,((Cy*Cz*sg)+(Cx*cg))/Cxz]])

            z1 = np.zeros((3,9))
            z2 = np.zeros((3,3))
            z3 = np.zeros((3,6))
            T1 = np.hstack((t,z1))
            T2 = np.hstack((z2,t,z3))
            T3 = np.hstack((z3,t,z2)) 
            T4 = np.hstack((z1,t))

            T = np.vstack((T1,T2,T3,T4))     #Transformation Matrix in local coordinate system

            transT = np.transpose(T)
            
            #Local Stiffeness Matrix in Global Coordinates 
            Kl = (transT@k)@T
            

            T_trans[member_no,:,:]= T  #Storing transformation matrix for each members                      
            
            ebc= np.hstack((ebc,Kl))       #stiffness matrix in global coordinates           
            
            self.__local_stiffness[member_no,:,:]= k
            KK_Local[eleno,:,:]= Kl
        
            qx= -m_L[mmm,0]
            qy= -m_L[mmm,1]
            qz= -m_L[mmm,2]
        

            if slabload == 0:
                # Local Forces from member load
                fx1= (qx*L)/2       #mmm= member with two nodes and m_L is member load
                fy1= (qy*L)/2
                fz1= (qz*L)/2
                mx1= 0                  #Torsion value Mx
                my1= (qz*L*L)/12
                mz1= (qy*L*L)/12   
        
                fx2= (qx*L)/2
                fy2= (qy*L)/2
                fz2= (qz*L)/2
                mx2= 0                  #Torsion value Mx
                my2= (qz*L*L)/12
                mz2= (qy*L*L)/12    

            
            if slabload == 1:
                
                if (self.__dlvdl.index == mem_name).any():
                    dl= ((self.__dlvdl [self.__dlvdl.index == mem_name])*1000).sum() # CHanging from kn to N
                    ll= ((self.__llvdl [self.__llvdl.index == mem_name])*1000).sum()
                    
                    fx1= (qx*L)/2       #mmm= member with two nodes and m_L is member load
                    fy1= ((qy*L)/2) + dl.Ra + ll.Ra
                    fz1= (qz*L)/2
                    mx1= 0                  #Torsion value Mx
                    my1= (qz*L*L)/12
                    mz1= ((qy*L*L)/12) + dl.Ma  + ll.Ma   
        
                    fx2= (qx*L)/2
                    fy2= ((qy*L)/2) + dl.Rb  + ll.Rb
                    fz2= (qz*L)/2
                    mx2= 0                  #Torsion value Mx
                    my2= (qz*L*L)/12
                    mz2= ((qy*L*L)/12) + dl.Mb + ll.Mb   


                else:
                    fx1= (qx*L)/2       #mmm= member with two nodes and m_L is member load
                    fy1= (qy*L)/2
                    fz1= (qz*L)/2
                    mx1= 0                  #Torsion value Mx
                    my1= (qz*L*L)/12
                    mz1= (qy*L*L)/12   
        
                    fx2= (qx*L)/2
                    fy2= (qy*L)/2
                    fz2= (qz*L)/2
                    mx2= 0                  #Torsion value Mx
                    my2= (qz*L*L)/12
                    mz2= (qy*L*L)/12        
            

            if self.__point_L==True:
                if (self.__point_loads.index == mem_name).any():
                    pl= (self.__point_loads [self.__point_loads.index == mem_name])

                    for pf in range (0,len(pl)):
                        pqx= 0
                        pqy= 0
                        pqz= 0

                        aaa= pl.iat[pf,2]
                        bbb= L- aaa
                        if pl.iat[pf,1] == 'x':
                            pqx= pl.iat[pf,0]


                        if pl.iat[pf,1] == 'y':
                            pqy= pl.iat[pf,0]*1000
                            PRay= -pqy* (bbb*bbb)*((3*aaa)+ bbb)/ (L*L*L) 
                            PRby= -pqy* (aaa*aaa)*((3*bbb)+ aaa)/ (L*L*L)
                            
                            PMay= -pqy* (bbb*bbb)*(aaa)/ (L*L) 
                            PMby= -pqy* (aaa*aaa)*(bbb)/ (L*L)

                            fy1= fy1 + PRay
                            fy2= fy2 + PRby
                            mz1= mz1 + PMay 
                            mz2= mz2 + PMby  

                        if pl.iat[pf,1] == 'z':
                            pqz= pl.iat[pf,0]*1000
                            PRaz= -pqz* (bbb*bbb)*((3*aaa)+ bbb)/ (L*L*L) 
                            PRbz= -pqz* (aaa*aaa)*((3*bbb)+ aaa)/ (L*L*L)
                            PMaz= -pqz* (bbb*bbb)*(aaa)/ (L*L) 
                            PMbz= -pqz* (aaa*aaa)*(bbb)/ (L*L)

                            fz1= fz1 + PRaz
                            fz2= fz2 + PRbz
                            my1= my1 + PMaz
                            my2= my2 + PMbz
        
            local_nodal_f=np.array([ fx1,fy1,fz1, mx1,-my1,mz1, fx2,fy2,fz2, mx2,my2,-mz2 ])                #Torsion value Mx
            
            #global_nodal_f is actually in Global Coordinates 
            self.__lnf[member_no,:,0] = local_nodal_f
            
            global_nodal_f= transT@local_nodal_f
 
        
            # local_forces is total equivalent nodal forces of all member load as per member order
            local_forces=np.hstack((local_forces,global_nodal_f))
            mmm = mmm + 1

                       
            for jj in range(tn): # tn is total nodes 
                if (cords[i,0]==nodes[jj,0] and cords[i,1]==nodes[jj,1] and cords[i,2]==nodes[jj,2] ):
                    ff= (jj+1)*6      #(jj*6)
                
                if (cords[h,0]==nodes[jj,0] and cords[h,1]==nodes[jj,1] and cords[h,2]==nodes[jj,2] ):
                    zz= (jj+1)*6    
 
        
            # Kg ig Global stiffness matrix with nodes order only
            # global_nodal_forces is equivalent nodal forces of member load in global coordinate in nodes order
            cc=0    
            for p in range(ff-6,ff,1) :
                dd=0
                #ff1=0
                for pp in range(ff-6,ff,1):
                    Kg[p,pp]= Kg[p,pp] + Kl[cc,dd]
                    dd=dd+1
                for ppp in range(zz-6,zz,1): 
                    Kg[p,ppp]= Kg[p,ppp] + Kl[cc,dd] 
                    dd=dd+1
                global_nodal_forces[p]= global_nodal_forces[p]+ global_nodal_f[cc]
                cc=cc+1
                
            cc=6
            for p in range(zz-6,zz,1) :
                dd=0
                for pp in range(ff-6,ff,1):
                    Kg[p,pp]= Kg[p,pp] + Kl[cc,dd]
                    dd=dd+1
                for ppp in range(zz-6,zz,1): 
                    Kg[p,ppp]= Kg[p,ppp] + Kl[cc,dd]
                    dd=dd+1
                global_nodal_forces[p]= global_nodal_forces[p]+global_nodal_f[cc]
                cc=cc+1      
            eleno= eleno +1 
            member_no = member_no+ 1
                      
        local_forces=local_forces.reshape((-1, 1))
        self.__K_Global= Kg
        #RC.k_locals_member= ebc         #k_locals_member_order
        self.__global_forces= global_nodal_forces       #global_nodal_forces- equi_nodalforces_nodes_order
        # self.__local_forces = local_forces              #local_forces_member_order
        self.__trans_mat= T_trans                            #T_trans_l
        self.__K_Local= KK_Local
        

    def __solution(self):           
        # ***This function caluclates the GLobal Forces and Displacements***
         
        forcevec= np.transpose(self.__forcesnodal.to_numpy().flatten())
        dispvec= np.transpose(self.__boundarycondition.to_numpy().flatten())  
        global_forces= self.__global_forces
        
        eq1= np.array([])    # Vector to find forces
        eq2= np.array([])    # Vector to find displacements
        tn = self.tn
        Kg = self.__K_Global

        if self.tm==1:
            if np.isin(dispvec,0).any():     
                if forcevec[3]==0 and forcevec[9] == 0:
                    dispvec[3]=0
                    dispvec[9]=0
                
        for i in range(tn*6):
        
            if dispvec[i]==0:
                eq1= np.hstack((eq1,i))  #storing position of known displacement
            else:
                eq2= np.hstack((eq2,i))  #storing position of unknown displacement


        # Forming new Equilibrium matrix's 
        ln = np.size(eq2)
        Kg1 = np.zeros([ln,ln])
        F2 = np.zeros(ln)
        local_member_nodal_f = np.zeros(ln)


        for i in range(ln):
            for j in range(ln):
                Kg1[i,j]= Kg[eq2[i].astype(int),eq2[j].astype(int)]
                F2[i] = forcevec[eq2[i].astype(int)]
            local_member_nodal_f[i] = global_forces[eq2[i].astype(int)]


        if (not np.any(eq1)) == False:
            D2 = (np.linalg.inv(Kg1))@(F2-local_member_nodal_f)
        else:
            D2 = np.zeros([len(eq2)])
        
        for i in range(ln):
            dispvec[eq2[i].astype(int)]=D2[i]
    
    
        F1 = global_forces+(Kg@dispvec) 
    

        forces= F1.reshape(tn,6)
        displac= dispvec.reshape(tn,6) 
        self.__GForces= forces                # in N and N-m
        self.__GDisplacement = displac           # in m and radian

    def __internal_forces_calculation(self):

        dis= self.__GDisplacement 
        mem_l= self.__member_details.loc[:,'xUDL':'zUDL']
        dis=dis.flatten().reshape((-1,1))
        nodes= self.__nodes_details.to_numpy()
        q= (mem_l.to_numpy())      #in Kn

        np.set_printoptions(precision = 3, suppress = True)  #surpressing exponential option while printing
        
        tm= self.tm
        BM_SFmat= [] #3D array to store BM and SF variation x= Member, y=value at some distance and z= xx,Vx,Vy,My,Mz
         
    
        mem_names = self.member_list
    
        member_nodes= np.empty([tm,2])
        
        if self.__slabload_there==1:
            sb= self.slab_pd.copy()
            sb.index= sb.Beam.to_list()


        for i in range(tm):
        
            en1 = self.__member_details.iloc[i,0]
            en2 = self.__member_details.iloc[i,1]
        
            pos1 = self.__nodes_details.index.get_loc(en1)
            pos2 = self.__nodes_details.index.get_loc(en2)
                                  
            n1= nodes[pos1,:]
            n2= nodes[pos2,:]        
            len_beam= np.sqrt(np.sum((n2-n1)**2))     # Length in mm
            L = len_beam
        
            d1= dis[pos1*6: (pos1*6)+6]
            d2= dis[pos2*6: (pos2*6)+6]
            ds = np.vstack((d1,d2))
            self.__ds.append(ds)

            #*** FOR INTERNAL FORCES in GLOBAL CORDINATE
            T_mat= self.__trans_mat[i,:,:].copy()
            kl_new= self.__local_stiffness[i,:,:]
            lf_new= self.__lnf[i,:,:]
            


            nodalForces= (kl_new@(T_mat@ds)) + lf_new

            nodalForces= nodalForces/1000        #converting into kN
            #------------------------------------------------------------------#

            member_nodes[i,0]= en1
            member_nodes[i,1]= en2
        

            parts = 2*L*100
            #------------------------------------------------------------------#
        
            xx=np.arange(0,L+0.001,L/parts)              
            jjj=np.ones(int(parts+1))
        
            # ** SF AND BM in LOCAL CORDINATE SYSTEM

            Vy= (nodalForces[1]+(q[i,1]*xx)).reshape((-1,1))
            
            Vx= ((nodalForces[0])*jjj).reshape((-1,1))
            Vz= (nodalForces[2]+(q[i,2]*xx)).reshape((-1,1))
            
            My = (nodalForces[4] + (nodalForces[2]*xx)+ (q[i,2]*((xx**2)/2))).reshape((-1,1)) 
            Mz = (nodalForces[5]-(nodalForces[1]*xx)-(q[i,1]*((xx**2)/2))).reshape((-1,1))
            Mx= np.ones(int(parts+1)).reshape((-1,1))* nodalForces[3]
            

            if self.__point_L==True: 
                
                if ((self.__point_loads.index == mem_names[i]).any()):
                    pl= (self.__point_loads [self.__point_loads.index == mem_names[i]])
                    pl= pl.sort_values(by=['Distance (m)'])
                    point_list= []
                    point_loads=[]
                    direction= []
                    for pf in range (0,len(pl)):
                        point_list.append(pl.iat[pf,2])
                        point_loads.append(pl.iat[pf,0])
                        direction.append(pl.iat[pf,1])

                    #point_list.append(L)
 
            
                    for val in range (len(point_list)):

                        PVy0= np.array([])
                        PMy0= np.array([])
                        PVz0= np.array([])
                        PMz0= np.array([])

                        xxxF= xx[xx<=point_list[val]]
                        xxxL= xx[xx>point_list[val]]

                        #fff= np.zeros(len(xxxF))
                        kkk= np.ones(len(xxxL))
                        P1= np.zeros(len(xxxF))
                        M1= np.zeros(len(xxxF))

                        if direction[val] == 'y':
                            Pforcey= point_loads[val]
                            
                        else:
                            Pforcey= 0

                        if direction[val] == 'z':
                            Pforcez= point_loads[val]
                        else:
                            Pforcez= 0

                        P2y= Pforcey*kkk
                        P2z= Pforcez*kkk

                        M2y= P2z* (point_list[val] - xxxL)
                        M2z= P2y* (point_list[val] - xxxL)

                        PVy= np.hstack((P1,P2y))  
                        PMy1 = np.hstack((M1,M2y))
                            
                        PVz= np.hstack((P1,P2z))  
                        PMz1 = np.hstack((M1,M2z))



                        PVy0= np.hstack((PVy0,PVy))
                        PMz0= np.hstack((PMz0,PMz1))

                        PVz0= np.hstack((PVz0,PVz))
                        PMy0= np.hstack((PMy0,PMy1))
                    
                        Vy= Vy + PVy0.reshape((-1, 1))   
                        Vz= Vz + PVz0.reshape((-1, 1))
                        Mz= Mz + PMz0.reshape((-1, 1))
                        My= My + PMy0.reshape((-1, 1))


            if self.__slabload_there==1:
                if (sb.index == mem_names[i]).any():
                    beam_no= (sb [sb.index == mem_names[i]])
                    beam_name= beam_no.index.values[0]
                    

                    for k in range (len(beam_no)):
                        H= beam_no.Height_l.values[k]
                        Lo= beam_no.L.values[k]

                        if len(beam_no)> 1:
                            w= (self.load_combo.iat[0,0]*((-self.__concrete_densitySlab*(sb.slab_t[beam_name].to_list()[k])/1000) + (sb.FF[beam_name].to_list()[k])+ (sb.WP[beam_name].to_list()[k])))      #kN/m2
                            wl= (self.load_combo.iat[0,1]*sb.LL[beam_name].to_list()[k]) #kN/m2
                        else:
                            w= (self.load_combo.iat[0,0]*((-self.__concrete_densitySlab*(sb.slab_t[beam_name])/1000) + (sb.FF[beam_name])+ (sb.WP[beam_name])))      #kN/m2
                            wl= (self.load_combo.iat[0,1]*sb.LL[beam_name]) #kN/m2                            

                        wf= w+wl

                        rat= (H/(Lo/2))

                        # FLOOR/SLAB LOAD Calulation  
                        if beam_no.Type_loading.values[k] == 1:     # Traingular Loading 
                            split_points= beam_no.LD4.tolist ()
                            posi= np.abs(xx - split_points[0][1]).argmin()
                            pv= split_points[0][1] 
                            if xx[posi] > split_points[0][1]:
                                posi= posi-1

                            # split position
                            xx1= xx[0:posi+1]
                            xx2= xx[posi+1:]
                            xx3= xx2[-1]- xx2
                            


                            Vy1= ((wf*xx1*xx1*(rat)/2))          
                            Vy2= ( (wf*pv*pv/2) + ((wf*H*H/2)-(wf*xx3*xx3*(rat)/2)))  


                            Mz1 = (-(wf*xx1*xx1*xx1/6))          
                            Mz2 =  -(((wf*H*xx2[-1]/2)*(xx2-pv)) +   ((wf*xx3*xx3/2)* (xx3/3)))
                            #(wf*H*H*(xx2-H)/6)+(((wf*H*H/2)*((xx2-(4*H/3))))-((wf*xx3*xx3/2)*(xx3/3)))        

                            Vy_vdl= np.hstack((Vy1,Vy2)).reshape((-1, 1))
                            Mz_vdl= np.hstack((Mz1,Mz2)).reshape((-1, 1))
                        
                            Vy= Vy + Vy_vdl
                            Mz= Mz + Mz_vdl

                        if beam_no.Type_loading.values[k] == 2:      # "Trapezoidal Loading"
                            
                            sp1= beam_no.LD1.tolist()
                            sp2= beam_no.LD2.tolist()
                            posi1= np.abs(xx - sp1[0][1]).argmin() 
                            posi2= np.abs(xx - sp2[0][1]).argmin()
                            pv2= sp2[0][1]
                            pv1= sp1[0][1]
                            if xx[posi1] > sp1[0][1]:
                                posi1= posi1-1

                            if xx[posi2] > sp2[0][1]:
                                posi2= posi2-1


                            # split position
                            xx1= xx[0:posi1+1]
                            xx2= xx[posi1+1:posi2+1]
                            xx4= xx[posi2+1:]
                            xx3= xx4[-1]- xx4

                            Vy1= ( (wf*xx1*xx1/2))    
                            Vy2= ( (wf*H*(xx2-(H/2)))) 
                            Vy3= (( (wf*H*(pv2-(H/2)))) +  ((wf*H*H/2)-(wf*xx3*xx3/2)))
 
                            Mz1 = -((wf*xx1*xx1*xx1/6)) 
                            Mz2 = -(((wf*H*(H/2))*(xx2-(2*H/3))) + (((xx2-pv1)**2)*wf*H/2 ))    
                            Mz3 =  -(((wf*H*(xx4[-1]+(pv2-pv1))/2)*(xx4-(xx4[-1]/2))) + ((wf*xx3*xx3/2)* (xx3/3)))
                            #((wf*pv1*pv1/2)*(xx4-(2*pv1/3))) + ((pv2-pv1)*wf*H*(xx4-(xx4[-1]/2)))       (((wf*H*xx2[-1]/2)*(xx2-pv)) +   ((wf*xx3*xx3/2)* (xx3/3)))   #.reshape((-1, 1))


                            Vy_vdl= np.hstack((Vy1,Vy2,Vy3)).reshape((-1, 1))
                            Mz_vdl= np.hstack((Mz1,Mz2,Mz3)).reshape((-1, 1))
            
                            Vy= Vy + Vy_vdl
                            Mz= Mz + Mz_vdl


        #---------------------------------------------------------------------#
            xx=xx.reshape((-1,1))
            mem_BMSF_val=np.hstack((xx,Vx,Vy,Vz,Mx,My,Mz))      
            BM_SFmat.append(np.around(mem_BMSF_val.astype(np.double),3))

        self.__member_nodes_inorder=member_nodes
        
        self.__SF_BM= BM_SFmat              # Shear Force and Bending Moment DATA 

    def __internalAll(self):
        dis= self.__GDisplacement 
        mem_l= self.__member_details.loc[:,'xUDL':'zUDL']
        dis=dis.flatten().reshape((-1,1))
        nodes= self.__nodes_details.to_numpy()
        q= (mem_l.to_numpy())      #in Kn

        np.set_printoptions(precision = 3, suppress = True)  #surpressing exponential option while printing
        
        tm= self.tm
        #3D array to store BM and SF variation x= Member, y=value at some distance and z= xx,Vx,Vy,My,Mz
         
    
        mem_names = self.member_list
    
        
        if self.__slabload_there==1:
            sb= self.slab_pd.copy()
            sb.index= sb.Beam.to_list()
        else:
            sb= None

        self.BM_SFmat1, self.member_nodes1, self.ds1 = ray.get([cal_internalF.remote(self.__member_details.iloc[i,:], self.__nodes_details.iloc[i,:], nodes, dis, self.__trans_mat[i,:,:], self.__local_stiffness[i,:,:], self.__lnf[i,:,:] , q , mem_names, i, self.__point_L , self.__point_loads, self.__slabload_there, sb, self.load_combo, self.__concrete_densitySlab) for i in range(tm)])


    @ray.remote
    def __defCal(E,I, d1,d2,L, bm):
    
        EI= E*I             #N-m2  --- E= N/m2,  I = m4
        bm=(bm*1000)/EI     # COnverting Bending Moment  from kNm to Nm and dividing by EI
        n=int(2*L*100) 

        h = (L-0) / n

        # Get A
        A = np.zeros((n+1, n+1))
        A[0, 0] = 1
        A[n, n] = 1
        for i in range(1, n):
            A[i, i-1] = 1
            A[i, i] = -2
            A[i, i+1] = 1
        
        # Get b
        ff= bm*(h**2)                        # NEW LINE 
        ff = ff.reshape((-1,1))

        ff[0]=d1
        ff[-1]=d2

        y = np.linalg.solve(A.astype(np.double), ff.astype(np.double))

        ymm=(y*1000).flatten()
        ymm=np.around(ymm,3)
        defl= ymm

        return defl


    def __CalDeflections(self):

        L= self.len_beam.copy() 
        nodes= self.__nodes_details.to_numpy()
        Framex3D= [] 
        Framey3D= [] 
        Framez3D= [] 

        def_mat= [] 
        def_mat_G= [] 

        def_shape= [] 

        for i in range (self.tm):

            Iz = self.__member_details.iat[i,9]
            Iy = self.__member_details.iat[i,10]
            J = self.__member_details.iat[i,11] 
            E = self.__member_details.iat[i,12]

            SF_BM= self.__SF_BM[i]
            ds= self.__ds[i]
            xx= SF_BM[:,0].reshape((-1,1))
            Mx= SF_BM[:,4]
            My= SF_BM[:,5]
            Mz= SF_BM[:,6]
            en1 = self.__member_details.iat[i,0]
            en2 = self.__member_details.iat[i,1]

        
            pos1 = self.__nodes_details.index.get_loc(en1)
            pos2 = self.__nodes_details.index.get_loc(en2)
                                  
            n1= nodes[pos1,:]
            n2= nodes[pos2,:] 
            
            parts= 2*L[i]*100

            EE= [E,E,E,E,E,E]
            II= [J,Iz,Iy]

            d1= [0,0,0,ds[0],ds[1],ds[2]]
            d2= [0,0,0,ds[6],ds[7],ds[8]]
            LL= [L[i],L[i],L[i],L[i],L[i],L[i]]
            BM= [-Mx, -Mz, -My] 

            II2= [J,Iz,Iy]
            BM2= [-Mx, -Mz, -My]

            if n1[0]==n2[0] and n1[1]==n2[1]:
                II2= [Iy,Iz,J]
                BM2= [-My,-Mz,Mx]   

            if n1[0]==n2[0] and n1[2]==n2[2]:
                II2= [Iz,J,Iy]
                BM2= [Mz,-Mx,My] 
            
            II.extend(II2)
            BM.extend(BM2)

            result = ray.get([self.__defCal.remote(EE[kk],II[kk],d1[kk],d2[kk],LL[kk],BM[kk]) for kk in range(6)])

            defLmem= np.hstack((xx,result[0].reshape((-1,1)),result[1].reshape((-1,1)),result[2].reshape((-1,1))))
            #result2 = ray.get([self.__defCal.remote(EE[kk],II[kk],d1[kk],d2[kk],LL[kk],BM[kk]) for kk in range(3)])


            DefG= np.hstack((xx, result[3].reshape((-1,1)),result[4].reshape((-1,1)),result[5].reshape((-1,1))))


            x1=np.linspace(n1[0],n2[0],int(parts +1) )
            y1=np.linspace(n1[1],n2[1],int(parts +1))
            z1=np.linspace(n1[2],n2[2],int (parts +1))

            dis_x= (DefG[:,1]/1000) + x1       #Deflected position in global coordinates in x,y,z direcion i.e global displacement
            dis_y= (DefG[:,2]/1000) + y1       #Deflected position in global coordinates in x,y,z direcion i.e global displacement
            dis_z= (DefG[:,3]/1000) + z1       #Deflected position in global coordinates in x,y,z direcion i.e global displacement



            Framex=np.linspace(x1,dis_x,100)
            Framey=np.linspace(y1,dis_y,100)
            Framez=np.linspace(z1,dis_z,100)

            Framex3D.append(Framex)
            Framey3D.append(Framey)
            Framez3D.append(Framez)


            dis_x= dis_x.reshape((-1, 1))
            dis_y= dis_y.reshape((-1, 1)) 
            dis_z= dis_z.reshape((-1, 1)) 
            dis_xyz= np.hstack((dis_x,dis_y,dis_z))
            def_shape.append(dis_xyz)
        

            def_mat.append(defLmem)


            def_mat_G.append(DefG)


        self.__Deflections_local= def_mat         # DEFLECTION in local cordinate system
        self.__Deflections_G= def_mat_G       # DEFLECTION in GLOBAL cordinate system
        self.__deflected_shape= def_shape     # Shape of displacement in 3D cordinate

        self.__Framex3D= Framex3D           #Frame to animate 3D displacement              
        self.__Framey3D= Framey3D           #Frame to animate 3D displacement 
        self.__Framez3D= Framez3D           #Frame to animate 3D displacement 
      
    def __calMaxmemF(self):
        axial_forces= []
        maxforces_pd= pd.DataFrame()
        k=0
        row_node2= ["Max", "Max"]
        row_node3= ["+ve", "-ve"]
        cols_names= ["Fx (kN)", "Fy (kN)", "Fz (kN)", "Mx (kN-m)", "My (kN-m)", "Mz (kN-m)" ]

        for i in self.__SF_BM:
            max_matrix= np.zeros((2,7))
            max_matrix[0,:]= np.amax(i,axis= 0)
            max_matrix[1,:]= np.amin(i,axis= 0)

            if abs(max_matrix[0,1]) >= abs(max_matrix[1,1]):
                max_axial= max_matrix[0,1]
            else:
                max_axial= max_matrix[1,1]

            axial_forces.append(max_axial)    

            row_node1=[self.member_list[k],self.member_list[k]]

            row_node= [
                row_node1, 
                row_node2, 
                row_node3]

            maxmemForces= pd.DataFrame(max_matrix[:,1:],index= row_node, columns= cols_names)
            
            maxforces_pd= pd.concat([maxforces_pd,maxmemForces]) 
            k= k + 1
        
        self.axial_forces_pd= pd.DataFrame({"Axial Loads": axial_forces} ,index= self.member_list)

    
        if self.tm > 2:
            maxforces_pd.loc[:,:,'+ve']= maxforces_pd.loc[:,:,'+ve'].where(maxforces_pd.loc[:,:,'+ve']>=0, "NA") 
            
            maxforces_pd.loc[:,:,'-ve']= maxforces_pd.loc[:,:,'-ve'].where(maxforces_pd.loc[:,:,'-ve']<0, "NA")

        self.__maxF_pd= maxforces_pd

    def __EQS(self,direction, seismic_load_factor):
        FL= self.story_lumploads.copy() 
        Z=  self.__seismic_def['Z'].item()
        R=  self.__seismic_def['R'].item()
        I= self.__seismic_def['I'].item()
        Sag= self.__seismic_def['Sag'].item()
        if direction=='x' or direction=='-x':
            base_nodes= self.__nodes_details.loc[self.__nodes_details.y== min(self.__nodes_details.y)]
            d= max(base_nodes.x) - min(base_nodes.x)
        if direction=='z' or direction=='-z':
            base_nodes= self.__nodes_details.loc[self.__nodes_details.z== min(self.__nodes_details.z)]
            d= max(base_nodes.z) - min(base_nodes.z)

        h= max(self.__nodes_details.y) - min(self.__nodes_details.y)
        if self.__seismic_def['Time Period'].item() ==0:
            if self.__infillwall == False:
                T= 0.075*(h**0.75)
            else:
                T= (0.09*h)/np.sqrt(d)
            self.__seismic_def['Time Period']= T

        if Sag ==0:
            if self.__seismic_def['Soil Type'].item()==1: #Hard Soil
                if T <0.4:
                    Sag= 2.5
                if T>0.4 and T<4:
                    Sag= 1/T
                if T>4:
                    Sag= 0.25                    
            if self.__seismic_def['Soil Type'].item()==2: #Medium Soil
                if T <0.55:
                    Sag= 2.5
                if T>0.55 and T<4:
                    Sag= 1.36/T
                if T>4:
                    Sag= 0.34
            if self.__seismic_def['Soil Type'].item()==3: #Medium Soil
                if T <0.67:
                    Sag= 2.5
                if T>0.67 and T<4:
                    Sag= 1.67/T
                if T>4:
                    Sag= 0.42                    
            self.__seismic_def['Sag']= Sag

        self.__seismicD= self.__seismic_def.copy()

        
        Ah= (Z/2) * (Sag)/ (R/I)      # horizontal seismic coefficient
        self.__seismicD[["Seismic Acceleration"]] = Ah

        W= abs((FL['Self-Weight(kN)'] + (FL['Other-Dead-Loads'])).sum()) #Seismic Weight
        self.__seismicD[["Seismic Weight"]] = W

        Vb= (Ah*W)

        WiHi= (FL.iloc[:,0]+ FL.iloc[:,1])*FL.Ht*FL.Ht
        WH= WiHi.sum()
        Vi = (WiHi/WH)*Vb

        ND= self.nodes_detail[['Floor', 'Stiffness']]

        for i in range (1,len(ND.Floor.unique())):
            ND_f=  ND.loc[ND['Floor']==i]

            Stiff_ratio= ND_f['Stiffness']/ min(ND_f['Stiffness'])
            Avg_Vi= Vi[i]/Stiff_ratio.sum()
            nodal_seismic_forces= Avg_Vi*Stiff_ratio

            self.__nodal_S_F= nodal_seismic_forces              # Nodal forces

            if direction=='x':
                self.__forcesnodal.loc[nodal_seismic_forces.index,'Fx']= seismic_load_factor*nodal_seismic_forces*1000
            if direction=='-x':
                self.__forcesnodal.loc[nodal_seismic_forces.index,'Fx']= -nodal_seismic_forces*1000*seismic_load_factor
            if direction=='z':
                self.__forcesnodal.loc[nodal_seismic_forces.index,'Fz']= nodal_seismic_forces*1000*seismic_load_factor
            if direction=='-z':
                self.__forcesnodal.loc[nodal_seismic_forces.index,'Fz']= -nodal_seismic_forces*1000*seismic_load_factor
        Vi[0]= Vb
        self.__SeismicShear= pd.DataFrame({"Seismic Shear": Vi})
        self.__SeismicShear.index.name= "Floor"

        
    def __drift(self):
        self.__ASD= pd.DataFrame()

        ND= self.nodes_detail[['Floor']]

        Gdisp= pd.DataFrame(self.__GDisplacement[:,0:3], columns = ['Avg. ux (mm)','Avg. uy (mm)','Avg. uz (mm)'], index = [self.__nodes_details.index])*1000   #converting into mm from m
        
        floor= []
        y_uni= self.__nodes_details.y.unique()
        y_uni= np.insert(y_uni,0,y_uni[0])
        for i in range (len(ND.Floor.unique())):
            ND_f=  ND.loc[ND['Floor']==i]
            floor.append(i)
            Avg_Story_disp= Gdisp.loc[ND_f.index].mean(numeric_only=True).to_frame().T
            Avg_Story_disp[['Story Height']]= y_uni[i+1] - y_uni[i]
            self.__ASD= pd.concat([self.__ASD,Avg_Story_disp], axis=0)

        
        self.__ASD[["Drift x", "Drift y", "Drift z" ]] = 0

        self.__ASD.iloc[1:,4]= (self.__ASD.iloc[1:,0].values - self.__ASD.iloc[:-1,0].values)/ (self.__ASD.iloc[1:,3]*1000).values
        self.__ASD.iloc[1:,5]= (self.__ASD.iloc[1:,1].values - self.__ASD.iloc[:-1,1].values)/ (self.__ASD.iloc[1:,3]*1000).values
        self.__ASD.iloc[1:,6]= (self.__ASD.iloc[1:,2].values - self.__ASD.iloc[:-1,2].values)/ (self.__ASD.iloc[1:,3]*1000).values
        self.__ASD.index= floor
        self.__ASD.rename_axis('Floor', inplace= True)


    def __resetRC(self):
        self.__nodes_details= self.joint_details.copy()
        self.__member_details= self.mem_details.copy()     
        self.__forcesnodal= self.nodalforces.copy()  
        self.__boundarycondition= self.boundcond.copy() 
        
        

    def preP(self):
        """This function of :class:`StrucPy.RCFA.RCF` objects performs pre-processing for the analysis for a Reinforced Concrete Frame that user intend to analyse. It organizes, arranges and prepares the data's for analysis. This function should be called in order for analysis to take place. 

        :param: None
        """
        if self.__PreP_status== True:
            pass

        self.__PreP_status = True

        self.__nodes_arrangement_for_members()

        if self.tm < 300:
            self.__arrange_beam_column_nodes()

        if self.tm > 300:
            self.__arrange_all()


        if self.autoflooring == True:
            self.__autoflooring()


        if self.__slabload_there==1 :
            self.__slab_beam_load_transfer()


        self.joint_details= self.__nodes_details.copy()
        self.mem_details= self.__member_details.copy()     
        self.nodalforces= self.__forcesnodal.copy()  
        self.boundcond= self.__boundarycondition.copy()




    def RCanalysis(self):
        """This function of :class:`StrucPy.RCFA.RCF` objects performs analysis for a Reinforced Concrete Frame that user intend to analyse. It generates all the post analysis data. This function should be called first before any other function. 

        :param: None
        """
        if self.__PreP_status == False:
            raise Exception("Perform Pre Processing of the structure using method 'preP'")

        self.__Analysis_performed= True

        self.__properties()
        self.__member_detailing()


        if self.__slabload_there==1:
            self.__floorLoading()
        self.__tloads()
       
        status=0
        if self.__seismic_def_status == True:
            if self.load_combo.iloc[0,2] > 0:
                direction= 'x'
                seismic_load_factor= self.load_combo.iat[0,2]
                status= 1
    
            if self.load_combo.iloc[0,3] > 0:
                direction= '-x'
                seismic_load_factor= self.load_combo.iat[0,3]
                status= 1                    
                
            if self.load_combo.iloc[0,4] > 0:
                direction= 'z'
                seismic_load_factor= self.load_combo.iat[0,4]
                status= 1
                
            if self.load_combo.iloc[0,5] > 0:
                direction= '-z'    
                seismic_load_factor= self.load_combo.iat[0,5]
                status= 1

            if status==0:
                direction= 'x'
                seismic_load_factor= 0                    

            self.__EQS(direction, seismic_load_factor)            

        self.__stiffnessbeam()

        self.__solution()

        self.__internal_forces_calculation()
        
        # self.__internalAll()    Under maintainaince 

        self.__CalDeflections()

        self.__calMaxmemF()

        if self.__seismic_def_status == True:
            self.__drift()

        self.__effectiveLength()

    def model3D(self):
        """Returns a *3D figure* of :class:`StrucPy.RCFA.RCF` objects presenting the 3D model of a Reinforced Concrete Frame/Member that user intend to analyse. It plots all the members as per :param nodes_details & member_details: in their global coordinate system*. It can be used to verify the input parameter.

        :param: None
        :return: A plotly figure for a member of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: Figure
        """
        xx= self.__nodes_details.to_numpy()
        nodetext= self.__nodes_details.index.to_numpy()
        xxx= self.__cords_member_order.to_numpy()
        tmm=len(xxx)
        fig1= go.Figure()
        fig2= go.Figure()
        fig3= go.Figure()
        fig4= go.Figure()
        fig5= go.Figure()

        fig1.add_trace(go.Scatter3d(x=xx[:,2],y=xx[:,0],z=xx[:,1],mode='markers+text', text=nodetext,textposition="top right"))
        kk=0
        mem_index= self.member_list

        mtcs = np.array([[],[],[]]).T

        ftcs = np.array([[],[],[]]).T
        
        mem_text=[]
        floor_text=[]

        for i in range(0,tmm,2):
            fig2.add_trace(go.Scatter3d(x=xxx[i:i+2,2],y=xxx[i:i+2,0],z=xxx[i:i+2,1], mode='lines+text',      
                line=dict(
                        color="black",                # set color to an array/list of desired values
                        width=10),name= f"member {kk+1}" ))

            ax= xxx[i,2].item() 
            bx= xxx[i+1,2].item() 
            ay= xxx[i,0].item() 
            by= xxx[i+1,0].item() 
            az= xxx[i,1].item() 
            bz= xxx[i+1,1].item() 

            x_anno=((ax+bx)/2)
            y_anno=((ay+by)/2)
            z_anno=((az+bz)/2)
            
            mtc = np.array([[x_anno],[y_anno],[z_anno]]).T

            mtcs= np.vstack((mtcs,mtc))
            mem_text.append(f"{mem_index[kk]}")
            kk= kk+1

        fig4.add_trace(go.Scatter3d(x=mtcs[:,0],y=mtcs[:,1],z=mtcs[:,2],mode='text', text=mem_text,textposition="middle right"))

        
        if self.__slabload_there==1:
            sb= self.__slab_details
            for i in sb.index:

                n1= sb.at[i,'Node1']
                n2= sb.at[i,'Node2']
                n3= sb.at[i,'Node3']
                n4= sb.at[i,'Node4']
                node= self.__nodes_details.loc[[n1,n2,n3,n4]]
                z= np.empty([5,5]) 
                z[:,:]= node['y'].values[0]   
                fig3.add_trace(go.Surface(x=node['z'], y=node['x'], z=z,opacity=0.2,showscale=False))        

                x_an=node["z"].mean()
                y_an=node["x"].mean()
                z_an=node["y"].mean()
            
                ftc = np.array([[x_an],[y_an],[z_an]]).T

                ftcs= np.vstack((ftcs,ftc))
                floor_text.append(f"Floor {i}")
                

        fig5.add_trace(go.Scatter3d(x=ftcs[:,0],y=ftcs[:,1],z=ftcs[:,2],mode='text', text=floor_text,textposition="top center"))                


        f1 = [trace for trace in fig1.select_traces()]
        f2 = [trace for trace in fig2.select_traces()]
        f3 = [trace for trace in fig3.select_traces()]
        f4 = [trace for trace in fig4.select_traces()]
        f5 = [trace for trace in fig5.select_traces()]

        button1= [True if i < (len(f1)+ len(f2)+ len(f3)) else False for i in range((len(f1)+ len(f2)+ len(f3) + len(f4)+ len(f5) ))]

        button2= [False for _ in range(len(f1))] + [True for _ in range(len(f2))] + [False for _ in range(len(f3))] + [True for _ in range(len(f4))] + [False for _ in range(len(f5))]

        button3= [False for _ in range(len(f1))] + [True for _ in range(len(f2))] + [True for _ in range(len(f3))] + [False for _ in range(len(f4))] + [True for _ in range(len(f5))]


        Model=go.Figure(data=f1+f2+f3+f4+f5)

        Model.update_layout(
            scene=dict(
                xaxis=dict(type="-"),
                yaxis=dict(type="-"),
                zaxis=dict(type="-"),))

        Model.update_layout(scene = dict(
                    xaxis_title=' ',
                    yaxis_title=' ',
                    zaxis_title=' '),)

        Model.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   yaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   zaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
             ))
  
        Model.update_layout(
            updatemenus=[
                dict(
                    type = "buttons",
                    direction = "left",
                    buttons=list([
                        dict(
                            args=[{'visible': button1},],
                            label="RC Model",
                            method="update",
                        ),
                        dict(
                            args=[{'visible': button2},],
                            label="Member Ids",
                            method="update"
                        ),
                        dict(
                            args=[{'visible': button3},],
                            label="Floors IDs",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="middle"
                ),
            ]
        )

        Model.update_layout(height=800, width=1500)

        return (Model)



    def sfbmd(self, element):
        """Returns a *figure* of :class:`StrucPy.RCFA.RCF` objects presenting the Shear Force Diagram and Bending Moment Diagram for an element of a Reinforced Concrete Frame/Member. It plots the diagram for member in their *local coordinate system* i.e showing results for Y and Z direction. It can be saved in any format using Plotly methods. 

        :param element: Name/Index of a member for which shear force and bending moment diagram is to be plotted.
        :type element: int
        :return: A plotly figure for a member of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: Figure (Object)
        """

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        #pio.renderers.default = rend_type



        pos = self.member_list.index(element)
        xx = self.__SF_BM[pos][:,0].round(decimals = 3)
        sfy= self.__SF_BM[pos][:,2].round(decimals = 3)
        sfz= self.__SF_BM[pos][:,3].round(decimals = 3) 
        bmy= self.__SF_BM[pos][:,5].round(decimals = 3)
        bmz= self.__SF_BM[pos][:,6].round(decimals = 3)
        parts= len(xx)
        yyb= np.zeros(parts)

        po1= self.__member_nodes_inorder[pos,0]
        po2= self.__member_nodes_inorder[pos,1]


        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Shear Force Diagram (y- Direction)- SFy ", "Shear Force Diagram (z- Direction)- SFz ",
                             "Bending Moment Diagram (y- Direction)- BMy ", "Bending Moment Diagram (z- Direction)- BMz"))

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=1, col=1)      
        fig.add_trace(go.Scatter(x=[0,0], y=[0,sfy[0]], mode='lines', line=dict(color="red") ), row=1, col=1) 
        fig.add_trace(go.Scatter(x=xx, y=sfy,mode='lines', line=dict(color="red")),row=1, col=1)
        fig.add_trace(go.Scatter(x=[xx[-1],xx[-1]], y=[0,sfy[-1]],mode='lines', line=dict(color="red") ), row=1, col=1)

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=1, col=2)
        fig.add_trace(go.Scatter(x=[0,0], y=[0,sfz[0]],mode='lines', line=dict(color="red") ), row=1, col=2)
        fig.add_trace(go.Scatter(x=xx, y=sfz,mode='lines',line=dict(color="red")), row=1, col=2)
        fig.add_trace(go.Scatter(x=[xx[-1],xx[-1]], y=[0,sfz[-1]],mode='lines', line=dict(color="red") ), row=1, col=2)
             

        #fig.add_trace(go.Scatter(x=xx, y=bmy,mode='lines'),
        #      row=2, col=1)

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=2, col=1)
        fig.add_trace(go.Scatter(x=[0,0], y=[0,bmy[0]],mode='lines', line=dict(color="red") ), row=2, col=1)
        fig.add_trace(go.Scatter(x=xx, y=bmy,mode='lines',line=dict(color="red")), row=2, col=1)
        fig.add_trace(go.Scatter(x=[xx[-1],xx[-1]], y=[0,bmy[-1]],mode='lines', line=dict(color="red") ), row=2, col=1)


        #fig.add_trace(go.Scatter(x=xx, y=bmz,mode='lines'),
        #      row=2, col=2)

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=2, col=2)
        fig.add_trace(go.Scatter(x=[0,0], y=[0,bmz[0]],mode='lines', line=dict(color="red") ), row=2, col=2)
        fig.add_trace(go.Scatter(x=xx, y=bmz,mode='lines',line=dict(color="red")), row=2, col=2)
        fig.add_trace(go.Scatter(x=[xx[-1],xx[-1]], y=[0,bmz[-1]],mode='lines', line=dict(color="red") ), row=2, col=2)


        fig.update_layout(showlegend=False)
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=False)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[sfy[0], sfy[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[sfz[0], sfz[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[bmy[0], bmy[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[bmz[0], bmz[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=2)

        fig.update_layout(height=1200, width=1200, title_text= f"Shear Force & Bending Moment Diagram of Member {element}")

        # Update xaxis properties
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=1, col=1)
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=1, col=2)
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=2, col=1)
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=2, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="<b>Shear Force (kN)</b>",  row=1, col=1)   #layout_yaxis_range=[-(abs(min(sfy))+5),(abs(max(sfy))+5)],
        fig.update_yaxes(title_text="<b>Shear Force (kN)</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Bending Moment (kN-m)</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Bending Moment (kN-m)</b>", row=2, col=2)

        return(fig)


    def defL(self, element):
        """Returns a *figure* of :class:`StrucPy.RCFA.RCF` objects presenting the deflected shape of an element of a Reinforce Concrete Frame/Member. It plots the deflection of a member in their *local coordinate system*. It can be saved in any format using Plotly methods.

        :param element: Name/Index of a member for which local deflection diagram is to be plotted.
        :type element: int
        :return: A plotly figure for a member of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: Figure
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        defl= self.__Deflections_local
        pos = self.member_list.index(element)
        xx = defl[pos][:,0].round(decimals = 3)
        defx= defl[pos][:,1].round(decimals = 3)
        defy= defl[pos][:,2].round(decimals = 3)
        defz= defl[pos][:,3].round(decimals = 3) 

        parts= len(xx)
        yyb= np.zeros(parts)

        po1= self.__member_nodes_inorder[pos,0]
        po2= self.__member_nodes_inorder[pos,1]

        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=("Deflection (x- Direction)", "Deflection (y- Direction)",
                             "Deflection (z- Direction)"))

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=1, col=1)      
        fig.add_trace(go.Scatter(x=xx, y=defx, mode='lines', line=dict(color="red") ), row=1, col=1) 


        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=2, col=1)
        fig.add_trace(go.Scatter(x=xx, y=defy,mode='lines', line=dict(color="red") ), row=2, col=1)


        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=3, col=1)
        fig.add_trace(go.Scatter(x=xx, y=defz,mode='lines', line=dict(color="red") ), row=3, col=1)

        fig.update_layout(showlegend=False)
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=False)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[sfy[0], sfy[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[sfz[0], sfz[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[bmy[0], bmy[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=3, col=1)


        fig.update_layout(height=1200, width=1200, title_text= f"Deflection of a Member {element} in Local Coordinate System")
        
        # Update xaxis properties
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=1, col=1)
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=2, col=1)
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=3, col=1)

        # Update yaxis properties
        fig.update_yaxes(title_text="<b>Deflection(mm)</b>",  row=1, col=1)   
        fig.update_yaxes(title_text="<b>Deflection(mm)</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Deflection(mm)</b>", row=3, col=1)
        return(fig)



    def defG(self, element):

        """Returns a *figure* of :class:`StrucPy.RCFA.RCF` objects presenting the deflected shape of an element of a Reinforced Concrete Frame/Member. It plots the deflection of a member in their *Global Coordinate System*. It can be saved in any format using Plotly methods.
         
        :param element: Name/Index of a member for which global deflection diagram is to be plotted.
        :type element: int
        :return: A plotly figure for a member of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: Figure
        """  
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        defl= self.__Deflections_G
        pos = self.member_list.index(element)
        xx = defl[pos][:,0].round(decimals = 3)
        defx= defl[pos][:,1].round(decimals = 3)
        defy= defl[pos][:,2].round(decimals = 3)
        defz= defl[pos][:,3].round(decimals = 3) 

        parts= len(xx)
        yyb= np.zeros(parts)

        po1= self.__member_nodes_inorder[pos,0]
        po2= self.__member_nodes_inorder[pos,1]


        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=("Deflection (x- Direction)", "Deflection (y- Direction)",
                             "Deflection (z- Direction)"))

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=1, col=1)      
        fig.add_trace(go.Scatter(x=xx, y=defx, mode='lines', line=dict(color="red") ), row=1, col=1) 


        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=2, col=1)
        fig.add_trace(go.Scatter(x=xx, y=defy,mode='lines', line=dict(color="red") ), row=2, col=1)


        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=3, col=1)
        fig.add_trace(go.Scatter(x=xx, y=defz,mode='lines', line=dict(color="red") ), row=3, col=1)

        fig.update_layout(showlegend=False)
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=False)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[sfy[0], sfy[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[sfz[0], sfz[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[0, 0],
            #y=[bmy[0], bmy[-1]],
            mode="markers+text",
            text=[f"Node {po1}", f"Node {po2}"],
            textposition="bottom center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=3, col=1)


        fig.update_layout(height=1200, width=1200, title_text=f"Deflection Of a Member {element} in Global Coordinate System")
        
        # Update xaxis properties
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=1, col=1)
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=2, col=1)
        fig.update_xaxes(title_text= "<b>Distance from node to node (m)</b>", row=3, col=1)

        # Update yaxis properties
        fig.update_yaxes(title_text="<b>Deflection(mm)</b>",  row=1, col=1)   
        fig.update_yaxes(title_text="<b>Deflection(mm)</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Deflection(mm)</b>", row=3, col=1)
        return(fig)

    def aniDef(self):
        """Returns a *figure* of :class:`StrucPy.RCFA.RCF` objects presenting the animation of the deflected shape of a Reinforced Concrete Frame. It shows the deflection animation of a frame in their *Global Coordinate System*. It can be saved in any format using Plotly methods.
         
        :param: None
        :return: A plotly figure for a member of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: Figure 
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        x1=self.__Framez3D
        y1=self.__Framex3D
        z1=self.__Framey3D
        n_frames=100

        frames = []
        for j in range (n_frames): 

            fig = go.Figure()
        
            fig.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   yaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   zaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                ))     
            camera = dict(eye=dict(x=0., y=2.5, z=0.))
            fig.update_layout(scene_camera=camera) 
            
            for i in range(self.tm):
                xx1= x1[i]
                yy1= y1[i]
                zz1= z1[i]
                fig.add_trace(go.Scatter3d(x=xx1[j,:],y=yy1[j,:],z=zz1[j,:], mode='lines',      
                    line=dict(
                        # set color to an array/list of desired values
                        width=5)))
            frames.append({'data':copy.deepcopy(fig['data']),'name':f'frame{j+1}'})

        fig.update(frames=frames)
        updatemenus = [dict(
                buttons = [
                    dict(
                        args = [None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True}],
                        label = "Play",
                        method = "animate"
                        ),
                    dict(
                        args = [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                        label = "Pause",
                        method = "animate"
                        )
                ],
                direction = "left",
                pad = {"r": 10, "t": 87},
                showactive = False,
                type = "buttons",
                x = 0.1,
                xanchor = "right",
                y = 0,
                yanchor = "top"
            )]  

        sliders = [dict(steps = [dict(method= 'animate',
                              args= [[f'frame{k}'],                           
                              dict(mode= 'immediate',
                                   frame= dict(duration=400, redraw=True),
                                   transition=dict(duration= 0))
                                 ],
                              label=f'{k+1}'
                             ) for k in range(n_frames)], 
                        active=0,
                        transition= dict(duration= 0 ),
                        x=0, # slider starting position  
                        y=0, 
                        currentvalue=dict(font=dict(size=12), 
                                  prefix='frame: ', 
                                  visible=True, 
                                  xanchor= 'center'
                                 ),  
                        len=1.0) #slider length
                ]
        fig.update_layout(width=800, height=800,  
                  
                  updatemenus=updatemenus,
                  sliders=sliders)
  

        fig.update_layout(height=1200, width=1200)

        return (fig)

    def def3D(self):

        """Returns a *figure* of :class:`StrucPy.RCFA.RCF` objects plots the deflected shape of a Reinforced Concrete Frame. It shows the deflection of a frame in their *Global Coordinate System*. It can be saved in any format using Plotly methods.
         
        :param: None
        :return: A plotly figure for a member of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: Figure
        """  
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        xx= self.__nodes_details.to_numpy()
        xxx= self.__cords_member_order.to_numpy()
        xxx1= self.__deflected_shape
        tmm=len(xxx)
        Model = go.Figure()
        Model.add_trace(go.Scatter3d(x=xx[:,2],y=xx[:,0],z=xx[:,1],mode='markers'))
        for i in range(0,tmm,2):
            Model.add_trace(go.Scatter3d(x=xxx[i:i+2,2],y=xxx[i:i+2,0],z=xxx[i:i+2,1], mode='lines',      
                line=dict(
                        color="black",                # set color to an array/list of desired values
                        width=10)))
        
        Model.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   yaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   zaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
             ))

        for i in range(self.tm):
            Model.add_trace(go.Scatter3d(x=xxx1[i][:,2],y=xxx1[i][:,0],z=xxx1[i][:,1], mode='lines',      
                line=dict(
                        color="red",                # set color to an array/list of desired values
                        width=10)))

        Model.update_layout(height=800, width=800, title_text=f"Deflection of Structure")
        return(Model)

    def reactions(self):     

        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` objects presenting nodal forces and moments of a Reinforced Concrete Frame. It shows the forces and moments in *Global Coordinate System*. Forces are displayed in KiloNewton (kN) and moments are displyed in KiloNewton-Meter(kN-m).
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  
  
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        GForces= self.__GForces.copy()
        GForces= GForces /1000       #converting into kN from N
        GForces = np.round(GForces.astype(np.double),3)
        Forcepd = pd.DataFrame(GForces, columns = ['Fx (kN)','Fy (kN)','Fz (kN)','Mx (kN-m)','My (kN-m)','Mz (kN-m)'], index = self.__nodes_details.index.to_list())  

        react= Forcepd.loc[self.baseN.index] 
        return react.sort_index()

    def Gdisp(self):        # Returns Diaplacement GLOBAL
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` objects presenting nodal translational and rotational displacements of a Reinforced Concrete Frame. It shows the displacements in *Global Coordinate System*. Translational displacements are displayed in milimeter(mm) and rotational displacements are displayed in radians(rad).
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        dispmat1= self.__GDisplacement.copy()
        dispmat1[:,0:3] = (dispmat1[:,0:3])*1000    # converting into mm
        dispmat1 = np.round(dispmat1.astype(np.double),3)
        Disppd = pd.DataFrame(dispmat1, columns = ['ux (mm)','uy (mm)','uz (mm)','rx (rad)','ry (rad)','rz (rad)'], index = self.__nodes_details.index.to_list())
        return Disppd
    
    def GlobalK(self):
        """Returns a *Numpy 2D Array* of :class:`StrucPy.RCFA.RCF` objects presenting a combined global stiffness matrix of a Reinforced Concrete Frame. Global stiffness matrix has been formed as per the order of nodal number/name in ascending order.
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        return self.__K_Global

    def LocalK(self,element= None):
        """Returns a *Numpy 2D Array or List of Numpy 2D Array* of :class:`StrucPy.RCFA.RCF` objects presenting a local stiffness matrix of a member or all members in global coordinate system of a Reinforced Concrete Frame.

        Returns list of 2D array when no argument is passed, local stiffness matrix of all the members in global coordinate system.

        Returns 2D array when argument is passed, local stiffness matrix of a members in global coordinate system.
         
        :param element: Optional, Default as None (Stiffness of all the members is passed). If particular member stiffness matrix is required, Name/Index of a member for which local stiffness matrix in global coordinate system is required must be passed. 
        :type element: int, Default as None 
        :return: A 2D numpy array or list of 2D numpy array of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: numpy.array/ list
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        if element==None:
            return self.__K_Local
        else:
            pos = self.member_list.index(element)
            return self.__K_Local[pos, :, :]


    def beamsD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` objects presenting details of all the beam elements in a reinforced concrete frame. It shows the nodes at which beam is connected, story of the beam, continuity at nodes, length in meters(m), cross-section in millimeters(mm), area in meters(m) moment of inertia in quartic meters(m\ :sup:`4`) and relative stiffness in cubic meters (m\ :sup:`3`) of beams. 
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  
        bd= self.beams_detail.copy()

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")       
        return(bd)

    def columnsD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` objects presenting details of all the column elements in a reinforced concrete frame. It shows the nodes at which column is connected, story of the column, length in meters(m), cross-section in millimeters(mm), area in meters(m\ :sup:`2`) moment of inertia in quartic meters(m\ :sup:`4`), relative stiffness in cubic meters (m\ :sup:`3`), effective length coefficient, effective length in meters(m) and type of columns (Pedestal, Short or Long). 
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  
        cd= self.columns_detail.copy()

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        return(cd) 

    def nodesD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` objects presenting details of all the nodes/joints in a reinforced concrete frame. It shows the floor of the nodes, height of the node from base in meters, beams name/index connected to the nodes, columns name/index connected to the nodes, and stiffness of the nodes. 
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  
        nd= self.nodes_detail.copy()

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        return(nd) 
    
    def Sdrift(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` objects presenting story drift of  a reinforced concrete frame. 
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
            

        if self.__seismic_def_status == False:
            return ("Seismic analysis has not been used")
        else:
            return (np.round(self.__ASD,5))
    
    def seismicS(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` objects presenting details of Base and Story Shear of a reinforced concrete frame in case of seismic analysis.
         
        :param: None
        :return: A DataFrame of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: pd.DataFrame
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
            

        if self.__seismic_def_status== False:
            return ("Seismic analysis has not been used")
        else:
            return (self.__SeismicShear)
        
    def memF(self):
        """Returns a *List* of 2D Numpy Array of :class:`StrucPy.RCFA.RCF` objects presenting detail of members (beams and columns) forces. 
        
        First column represents distance from lower number node to higher number node (2 nodes at end of member).
        Second column represents axial force in x-direction of the member in local coordinate system.
        Third column represents shear force in y-direction of the member in local coordinate system.
        Fourth column represents shear force in z-direction of the member in local coordinate system.
        Fifth column represents torsional moment in x-direction of the member in local coordinate system.
        Sixth column represents bending moment in y-direction of the member in local coordinate system.
        Seventh column represents bending moment in z-direction of the member in local coordinate system.

        Note: List position represents elements/members of reinforced concrete frame. All the members are arranged in acending order of their index number containing 2D numpy array respectively. 
         
        :param: None
        :return: A *List* of 2D  numpy array of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: numpy array
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
            
        
        return (self.__SF_BM)
    

    def MaxmemF(self):
        """Returns a *List* of 2D Numpy Array of :class:`StrucPy.RCFA.RCF` objects presenting detail of members (beams and columns) forces. 
        
        First column represents distance from lower number node to higher number node (2 nodes at end of member).
        Second column represents axial force in x-direction of the member in local coordinate system.
        Third column represents shear force in y-direction of the member in local coordinate system.
        Fourth column represents shear force in z-direction of the member in local coordinate system.
        Fifth column represents torsional moment in x-direction of the member in local coordinate system.
        Sixth column represents bending moment in y-direction of the member in local coordinate system.
        Seventh column represents bending moment in z-direction of the member in local coordinate system.

        Note: x position in 3D Numpy array represents elements/members of reinforced concrete frame. All the members are arranged in acending order of their index number representing 3D numpy array respectively. 
         
        :param: None
        :return: A 3D numpy array of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: numpy array
        """  
        return (self.__maxF_pd)

    def defLD(self):
        """Returns a *List* of 2D Numpy Array of :class:`StrucPy.RCFA.RCF` objects presenting detail of deflection detail of members (beam and columns) in local coordinate system. 
        
        First column represents distance from lower number node to higher number node (2 nodes at end of member)
        Second column represents deflection in x-direction of the member in local coordinate system.
        Third column represents deflection in y-direction of the member in local coordinate system.
        Fourth column represents deflection in z-direction of the member in local coordinate system.

        Note: List position represents elements/members of reinforced concrete frame. All the members are arranged in acending order of their index number containg 2D numpy array respectively. 
         
        :param: None
        :return: A *List* of 2D numpy array of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *List* of 2D numpy array
        """  

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
            
        
        return (self.__Deflections_local)    
    
    def defGD(self):
        """Returns a *List* of 2D Numpy Array of :class:`StrucPy.RCFA.RCF` objects presenting detail of deflection detail of members (beam and columns) in global coordinate system. 
        
        First column represents distance from lower number node to higher number node (2 nodes at end of member)
        Second column represents deflection in x-direction of the member in global coordinate system.
        Third column represents deflection in y-direction of the member in global coordinate system.
        Fourth column represents deflection in z-direction of the member in global coordinate system.

        Note: List position represents elements/members of reinforced concrete frame. All the members are arranged in acending order of their index number containg 2D numpy array respectively. 
         
        :param: None
        :return: A *List* of 2D numpy array of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *List* of 2D numpy array
        """  

        if self.__Analysis_performed== False:
            exit("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        return (self.__Deflections_G)
    
    def Mproperties(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the material properties of the members being used for the analysis of reinforced concrete frame. 
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* of panda
        """  

        return (self.__Mproperties)

    def floorD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the floor details of the reinforced concrete frame.  It shows the floor ID along with the beams at edges , floor thickness and floor loads.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  
        if self.__PreP_status == False:
            raise Exception("Perform Pre-Processing of the structure using method 'preP' to get complete floor details")

        if self.__slabload_there== False:
            return ("No Floor has been assigned")
        else:
            return (self.__slab_details)

    def seismicD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the seismic details of the reinforced concrete frame.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  
        cols= ["Zone Factor (Z)", "Importance Factor (I)", "Response Reduction Factor (R)",  "Design Accelaration Coefficient (Sag) based on Soil Type", "Damping (%)", "Soil Type", "Time Period (sec)", "Seismic Acceleration (Ah)", "Seismic Weight (kN)" ]

        if self.__seismic_def_status== True:
            if (self.load_combo.iloc[0, 2:] >0).any():
                self.__seismicD.columns= cols
                return (self.__seismicD)

    def changeFL(self, floor=None,thickness= None, LL=None, FF=None, WP=None, delf=None ):
        """This non returing function of :class:`StrucPy.RCFA.RCF` objects performs change in floor/slab. If 'floor' argument is none changes are made to every floor/slab and if floor number is passed in argument 'floor', changes will be made in that particular floor/slab.

        1. Changes can be performed on all floor together.
        2. Changes can be performed on only selected floor together.   
        3. Thickness of floors/slabs can be changed.
        4. Live Load on floors/slabs can be changed.
        5. Floor Finish Load on floors/slabs can be changed.
        6. Water Proffing Load on floors/slabs can be changed.
        7. Any particular floor or the group of floor can be deleted. 
        
        :param floor: Optional, Default as None (Changes to all floors/slabs).  It represents floor numbers/names on which chages has to be made. If changes has to be made on particular floor/slab or a group of floor/slab, number/name of the floors/slab has to be passed in argument 'floor'. 
        :type floor: int, list

        :param thickness: Optional, Default as None (No changes in thickness of floors/slabs).  It represents thickness of floor/slab in "millimeter(mm)".  
        :type thickness: int, list

        :param LL: Optional, Default as None (No changes in live load acting on the floors/slabs).  It represents represents the live load acting on the floors/slabs in "kiloNewton/square meter (kN/m2)".  
        :type LL: int, list

        :param FF: Optional, Default as None (No changes in floor finishing load acting on the floors/slabs).  It represents represents the floor finish load acting on the floors/slabs in "kiloNewton/square meter (kN/m2)".  
        :type FF: int, list 

        :param WP: Optional, Default as None (No changes in water proofing load acting on the floors/slabs).  It represents represents the water proofing load acting on the floors/slabs in "kiloNewton/square meter (kN/m2)".  
        :type WR: int, list  

        :param delf: Optional, Default as None (No deletion of any floor/slab).  It represents floor numbers/names which has to be deleted. If particular floor/slab or a group of floor/slab, number/name of the floors/slab has to be deleted, number/names of the floor has to passed in argument 'delf'.  
        :type delf: int, list   
        :return: None  
        """
        if self.__PreP_status == False:
            raise Exception("Perform Pre-Processing of the structure using method 'preP' before changing floor details")

        if floor==None:
            if thickness is not None:
                self.__slab_details[["Thickness(mm)"]]= thickness

            if FF is not None:
                self.__slab_details[["FF(kN/m2)"]]= FF

            if LL is not None:
                self.__slab_details[["LL(kN/m2)"]]= LL

            if WP is not None:
                self.__slab_details[["Waterproofing(kN/m2)"]]= WP

            if thickness== None and FF==None and LL==None and WP==None and delf==None:
                exit("Nothing has been passed to perform changes")

        if floor is not None:
            if floor.dtype in ["int32","int64"]:
                if thickness is not None:
                    self.__slab_details.loc[floor, ["Thickness(mm)"]]= thickness

                if FF is not None:
                    self.__slab_details.loc[floor, ["FF(kN/m2)"]]= FF

                if LL is not None:
                    self.__slab_details.loc[floor, ["LL(kN/m2)"]]= LL

                if WP is not None:
                    self.__slab_details.loc[floor, ["Waterproofing(kN/m2)"]]= WP
        
        if delf is not None:
            self.__slab_details.drop(delf, inplace = True)

        self.slab_pd= None
        self.__slab_beam_load_transfer()

        if self.__Analysis_performed== True:
            self.__resetRC()
        
        self.__Analysis_performed= False
        

    def changeLC(self, loadC):
        """This non returing function of :class:`StrucPy.RCFA.RCF` objects performs change in load combination without creating a new object for the analysis for a Reinforced Concrete Frame. 
        :ref:`InputExample:Load Combination Details` for more details.

        :param loadC: New load combination 'loadC' will be used for analysis. New analysis has to be performed by calling the method 'StrucPy.RCFA.RCF.RCanalysis()'.
        :type loadC: DataFrame
        :return: None
        """        
        self.load_combo= loadC.copy()
        self.load_combo.columns= ["Dead_Load","Live_Load","EQX","-EQx","EQZ","-EQZ"]
        self.load_combo.fillna(0,inplace=True)
        self.__Analysis_performed= False

    def changeFrame(self, member, node= None):
        """This non returing function of :class:`StrucPy.RCFA.RCF` object performs changes by deleting members and nodes of Reinforced Concrete Frame. It changes the frame model. 
        :ref:`InputExample:changeFrame` for more details.

        :param member: Member ID/ID's which has be deleted from the frame.
        :type member: int/list

        :param node: node ID/ID's which has be deleted from the frame.
        :type node: int/list
        :return: None
        """

        if not isinstance(member, (int, list)):
            raise Exception ("The member ID provided is of wrong type. It can only be int,float or list ")
        
        if isinstance(member, int):
            member_nodes_check= self.__member_details.index.isin([member])

            if member_nodes_check.any():
                self.__member_details.drop([member], inplace= True)
            else:
                raise Exception (f"These {member} member ID does not exist. " )
            
        if isinstance(member, list):

            member_nodes_check=    pd.Series(member).isin(self.__member_details.index)

            if member_nodes_check.all():
                self.__member_details.drop(member, inplace= True)

            else:
                raise Exception ("These nodes does not exist in member_details: ",  [i for i, val in enumerate(member_nodes_check) if not val] )
          


        if node is not None:
            if not isinstance(node, (int, list)):
                raise Exception ("The nodes ID provided is of wrong type. It can only be int,float or list ")

        if isinstance(node, int):
            nodes_check= self.__nodes_details.index.isin([node])

            if nodes_check.any():
                member_nodes_check= self.__member_details.loc[:,["Node1", "Node2"]].isin([node])

                if member_nodes_check.any().any():
                    raise Exception ("These node can not be deleted as it is being used by a member. First delete the member in order to remove the nodes")
                
                self.__nodes_details.drop([node], inplace= True)
            else:
                raise Exception (f"The {node} node ID does not exist. " )
            
        if isinstance(member, list):
            nodes_check=    pd.Series(node).isin(self.__nodes_details.index)

            if nodes_check.all():
                member_nodes_check= self.__member_details.loc[:,["Node1", "Node2"]].isin(node)

                if member_nodes_check.any().any():
                    raise Exception ("These node can not be deleted as it is being used by a member. First delete the member in order to remove the nodes")

                self.__nodes_details.drop(node, inplace= True)
            else:
                raise Exception ("These nodes does not exist in details: ",  [i for i, val in enumerate(nodes_check) if not val] ) 
            
        n1= self.__member_details.iloc[:,0:2].to_numpy()
        orphan_nodes_status = self.__nodes_details.index.isin(n1.flatten())
        
        orphan_nodes= [i for i, val in enumerate(orphan_nodes_status) if not val]

        self.__nodes_details.drop(orphan_nodes, inplace= True)

        self.__PreP_status= False
        self.__Analysis_performed= False

    def changeBoundcond(self, bound_conditions):
        """This non returing function of :class:`StrucPy.RCFA.RCF` object performs changes in boundary condition of Reinforced Concrete Frame. It completely  replaces the boundary existing boundary consition with the passed one. :ref:`InputExample:changeBoundcond` for more details.

        :param bound_conditions: Details of the new boundary condition to be used in analysis.
        :type bound_conditions: Dataframe
        :return: None
        """

        if not isinstance(bound_conditions, pd.DataFrame):
            raise Exception ("The bound_conditions provided is of wrong type. It can only be dataframe ")
        
        if len(bound_conditions.columns) != 6:
            raise Exception ("The boundary condition dataframe must contain 6 columns representing each degree of freedom in 3D space i.e. 'Trans x', 'Trans y', 'Trans z', 'Rotation x', 'Rotation y', 'Rotation z'. ")
        
        self.__bc_index = bound_conditions.index.isin(self.__nodes_details.index)

        if self.__bc_index.all():
            if len(bound_conditions)==self.node_list:
                self.__boundarycondition= bound_conditions.sort_index()
                self.__boundarycondition.columns= ["x","y","z","thetax","thetay","thetaz"]
            elif len(bound_conditions)!=self.node_list:             
                self.__boundarycondition = pd.DataFrame(np.ones([self.tn,6]),index=self.node_list,columns=["x","y","z","thetax","thetay","thetaz"])
                bound_conditions.sort_index(inplace=True)
                self.__boundarycondition.loc[bound_conditions.index]= bound_conditions.loc[:]
        else:
            raise Exception ("These nodes in boundary condition does not exist: ",  [i for i, val in enumerate(self.__bc_index) if not val] )

        self.__PreP_status= False
        self.__Analysis_performed= False

    def modelND(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the nodes of the reinforced concrete frame. It a parameter passed by user or generated by agrumunet `framegen`.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  
        return (self.__ndd)
    
    def modelMD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the members of the reinforced concrete frame. It a parameter passed by user or generated by agrumunet `framegen`.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  

        return (self.__mdd)
    
    def modelBCD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the boundary condition of the reinforced concrete frame. It a parameter passed by user or generated by agrumunet `framegen`.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  

        return (self._bcd)

class RCFenv():
    """
    This is a class to represent the analysis of 2D/3D reinforced concrete frame model with multiple load combinations. It form envelop for different load combinations. It also creates an object for the reinforced concrete frame model on which analysis is performed for every load combinations.


    :param nodes_details: A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the nodes and their coordinates in space. Check :ref:`InputExample:Nodes Details` for more details.
    :type nodes_details: DataFrame
    
    :param member_details: A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the members of reinforced concrete frame along with thier nodes, cross-section and loading details. Check :ref:`InputExample:Member Details` for more details.
    :type member_details: DataFrame
    
    :param boundcondition: A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the nodes/joints condition of reinforced concrete frame i.e. types of supports (fixed, hinged etc.) and joints conditions. Check :ref:`InputExample:Boundary Conditions` for more details.
    :type boundcondition: DataFrame
    
    :param framegen: A handle to the :class:`StrucPy.RCFA.RCF` that detects the number of bays and total length of required reinforced concrete frame along x- axis, z-axis and height along y-axis. Check :ref:`InputExample:framegen` for more details.
    :type framegen: DataFrame 

    :param forcesnodal: A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the nodes/joints forces of reinforced concrete frame, defaults to None (No nodel forces or moments). Check :ref:`InputExample:Nodal Forces Details` for more details.
    :type forcesnodal: DataFrame, optional
    
    :param slab_details:  A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the slabs/floor loads along with the nodes within which a slab/floor is formed, defaults to None (No floor load). Check :ref:`InputExample:Slab Details` for more details.
    :type slab_details: DataFrame, optional
    
    :param load_combo:  A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the load combination for the analysis of reinforced concrete frame. It takes the load factor respective to load case. Defaults to None (Only Dead Load case will be considered with load factor 1). Check :ref:`InputExample:Load Combination Details` for more details.
    :type load_combo: DataFrame, optional
    
    :param seismic_def:  A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the seismic load being applied to reinforced concrete frame. Defaults to None (No Seismic Load). Check :ref:`InputExample:Seismic Definition` for more details.
    :type seismic_def: DataFrame, optional
    
    :param properties: A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the properties of the members to be considered in analysis. Default to None (Concrete of Grade M25 ). Check :ref:`InputExample:Material Properties` for more details.
    :type properties: DataFrame, optional
    
    :param grade_conc: A handle to the :class:`StrucPy.RCFA.RCFenv` that detects the grade of concrete (like M25, M30) to be considered in analysis. Defaults to 25 N/mm2. Check :ref:`InputExample:Concrete Grade` for more details.
    :type grade_conc: float/int, optional
    
    :param self_weight:  A handle to the :class:`StrucPy.RCFA.RCFenv` that detects whether self weight of the members are to be considered in analysis or not. Defaults to True (Self weight considered). Check :ref:`InputExample:Self Weight` for more details.
    :type self_weight: Boolean, optional
    
    :param infillwall:  A handle to the :class:`StrucPy.RCFA.RCFenv` that detects whether infillwall has to be considered during the caculation of time period during seismic analysis (Applicable only for IS 1893: 2016 Part 1). Defaults to False (infillwall not considered). Check :ref:`InputExample:infillwall` for more details.
    :type infillwall: Boolean, optional
    
    :param autoflooring:  A handle to the :class:`StrucPy.RCFA.RCFenv` that detects whether the slab/floor load has to be generated automatically or not. It is highly usefull when dealing with large reinforced concrete framed structures. Defaults to False (Autoflooring not being done).  Check :ref:`InputExample:Autoflooring` for more details.
    :type autoflooring: Boolean, optional

    :param col_stablity_index:  A handle to the :class:`StrucPy.RCFA.RCFenv` that determines whether the column or frame is sway or not. It is highly usefull when dealing with large reinforced concrete framed structures. Defaults to 0.04 (IS456:2000).  Check :ref:`InputExample:Stability Index` for more details.
    :type col_stablity_index: Float/Int, optional
    """

    def __init__(self, nodes_details, member_details, boundarycondition, load_combo, framegen= None, forcesnodal=None, slab_details=None, seismic_def=None,self_weight= True, infillwall=False, autoflooring= False, properties= None, grade_conc= 25, col_stablity_index= 0.04):

        if framegen is not None:
            if not isinstance(framegen, pd.DataFrame):
                raise TypeError ("Type of 'framegen' must be DataFrame")    

            if len(framegen.columns) != 2:
                raise Exception ("framegen must have 2 columns: ['Number of bays', 'Total Length']")
            
            if len(framegen.index) != 3:
                raise Exception ("framegen must have 3 rows: ['Along length (x-axis) ', 'Along height (y-axis)'], 'Along width (z-axis)']")            

            lx= framegen.iat[0,1]
            ly= framegen.iat[1,1]
            lz= framegen.iat[2,1]
            nx= framegen.iat[0,0]
            ny= framegen.iat[1,0]
            nz= framegen.iat[2,0]
            x = np.linspace(0, lx, nx+1)
            y = np.linspace(0, ly, ny+1)
            z = np.linspace(0, lz, nz+1)


            zv,xv = np.meshgrid(z,x)

            xv= xv.flatten()
            zv= zv.flatten()

            zvv, yv = np.meshgrid(zv, y)

            xvv,yvv=  np.meshgrid( xv, y)

            cord_array= np.vstack((xvv.flatten(), yv.flatten(), zvv.flatten())).T
            nodes_details= pd.DataFrame(cord_array, columns= ['x','y','z'], index = [i for i in range (1,len (cord_array)+1)])


            total_members= (((nx* (nz+1))+ ((nx+1)* nz)) * ny) + (((nx+1)* (nz+1) * ny))
            mem_cords= np.empty([total_members,2])
            member_number= 0

            for ka in range (1,len(y)):
                y_val= y[ka]
                new_nodes1= nodes_details[nodes_details.y.isin([y_val])]
                for ia in x:
                    new_node2= new_nodes1[new_nodes1.x.isin([ia])]
                    node_ids= new_node2.index.to_list()
                    for ja in range (len(new_node2)-1):
                        mem_cords[member_number, 0] = node_ids[ja]
                        mem_cords[member_number, 1] = node_ids[ja+1]
                        member_number = member_number+1

                for ia in z:
                    new_node2= new_nodes1[new_nodes1.z.isin([ia])]
                    node_ids= new_node2.index.to_list()
                    for ja in range (len(new_node2)-1):
                        mem_cords[member_number, 0] = node_ids[ja]
                        mem_cords[member_number, 1] = node_ids[ja+1]
                        member_number = member_number+1

            nodes_details_usable= nodes_details.sort_values(by=['x', 'z']).copy()
            node_ids= nodes_details_usable.index.to_list()

            for ka in x:
                new_nodes1= nodes_details[nodes_details.x.isin([ka])]
                for ia in z:    
                    new_node2= new_nodes1[new_nodes1.z.isin([ia])]
                    node_ids= new_node2.index.to_list()
                    for ja in range (len(new_node2)-1):
                        mem_cords[member_number, 0] = node_ids[ja]
                        mem_cords[member_number, 1] = node_ids[ja+1]
                        member_number = member_number+1       
            
            mem_cords= mem_cords.astype(np.int64)
            member_details= pd.DataFrame(mem_cords, columns= ['Node1','Node2'], index = [i for i in range (1,len (mem_cords)+1)])

            member_details[['b', 'd']]= 500
            member_details[['xUDL', 'yUDL', 'zUDL']]= 0
            base_nodes= nodes_details[nodes_details.y.isin([y[0]])]

            boundarycondition_array= np.zeros ([len(base_nodes), 6])
            boundarycondition= pd.DataFrame(boundarycondition_array, columns= ["x","y","z","thetax","thetay","thetaz"], index = base_nodes.index)

        if not all(isinstance(i, pd.DataFrame) for i in [nodes_details, member_details, boundarycondition]):
            raise TypeError ("Type of the argument must be DataFrame")

        if len(member_details.columns)<7:
            raise Exception("MEMBER DETAILS must have 7 columns: ['Node1', 'Node2', 'b', 'd', 'xUDL', 'yUDL', 'zUDL'], First 4 columns are mandotory argument while last three loads can be left empty or with zero ")

        if len(member_details.columns)>7 :
            raise Exception("MEMBER DETAILS can have maximum of 7 columns: ['Node1', 'Node2', 'b', 'd','xUDL', 'yUDL', 'zUDL']")        
        
        if len(nodes_details.columns)!=3 :
            raise Exception("NODE DETAILS must have x,y and z coordinate: ['x', 'y', 'z']")
        
        if nodes_details.index.dtype not in ["int32","int64"]:
            raise TypeError("Node Number(Index) in 'nodes_details' must be 'int' type")

        if member_details.index.dtype not in ["int32","int64"]:
            raise TypeError("Member Numbe(Index) in 'member_details' must be 'int' type")
        
        if nodes_details.index.all()<1:
            raise NameError("Node Number(Index) in 'nodes_details' must be positive integer")

        if member_details.index.all()<1:
            raise NameError("Member Number(Index) in 'member_details' must be positive integer")
        
        

        member_details.columns = ['Node1', 'Node2', 'b', 'd','xUDL', 'yUDL', 'zUDL']
        nodes_details.columns = ['x', 'y', 'z']

        self.nodes_details= nodes_details.sort_index()      #self.joint_details to be used
        
        self.member_details= member_details.sort_index()  #self.mem_details to be used

        for i in range (len(self.member_details)):
                n1= self.member_details.iloc[i,0]
                n2= self.member_details.iloc[i,1]
                if (self.nodes_details.loc[n1]>self.nodes_details.loc[n2]).any():
                    n3= n2
                    n2=n1
                    n1= n3
                self.member_details.iloc[i,0]= n1
                self.member_details.iloc[i,1]= n2
        
        self.member_details.fillna(0,inplace=True)
        self.nodes_details.fillna(0,inplace=True)

        self.member_list= self.member_details.index.to_list()
        self.node_list= self.nodes_details.index.to_list()
        self.tn= self.nodes_details.shape[0]
        self.tm= self.member_details.shape[0]

        member_nodes_check= self.member_details.loc[:,["Node1", "Node2"]].isin(nodes_details.index)

        if member_nodes_check.all().all():
            pass
        else:
            raise Exception ("These nodes present in member details does not exist: ",  member_nodes_check[member_nodes_check["Node1"]==False]["Node1"] , member_nodes_check[member_nodes_check["Node2"]==False]["Node2"]  )
        
        depth_check= self.member_details.index[self.member_details["d"]>=0].to_list()


        if len(depth_check)!= self.tm:
            raise Exception ("The depth of some members has not been fixed: ")


        if forcesnodal is None:
            self.forcesnodal = pd.DataFrame(np.zeros([self.tn,6]),index=self.node_list,columns=["Fx","Fy","Fz","Mx","My","Mz"])
        
        if forcesnodal is not None:
            if not isinstance(forcesnodal, pd.DataFrame):
                raise TypeError ("Type of the 'forcesnodal' must be DataFrame")           
            
            self.__fv_index = forcesnodal.index.isin(nodes_details.index)

            if self.__fv_index.all():
                if len(forcesnodal)==self.node_list:
                    self.forcesnodal= forcesnodal.sort_index()
                    self.forcesnodal.columns= ["Fx","Fy","Fz","Mx","My","Mz"]
                elif len(forcesnodal)!=self.node_list:           
                    self.forcesnodal = pd.DataFrame(np.zeros([self.tn,6]),index=self.node_list,columns=["Fx","Fy","Fz","Mx","My","Mz"])
                    forcesnodal.sort_index(inplace=True)
                    self.forcesnodal.loc[forcesnodal.index]= forcesnodal.loc[:]
            else:
                raise Exception ("These nodes in nodal forces DataFrame does not exist: ",  [i for i, val in enumerate(self.__fv_index) if not val] )

        if len(boundarycondition.columns) != 6:
            raise Exception ("The boundary condition dataframe must contain 6 columns representing each degree of freedom in 3D space i.e. 'Trans x', 'Trans y', 'Trans z', 'Rotation x', 'Rotation y', 'Rotation z'. ")
        
        self.__bc_index = boundarycondition.index.isin(nodes_details.index)

        if self.__bc_index.all():
            if len(boundarycondition)==self.node_list:
                self.boundarycondition= boundarycondition.sort_index()
                self.boundarycondition.columns= ["x","y","z","thetax","thetay","thetaz"]
            elif len(boundarycondition)!=self.node_list:             
                self.boundarycondition = pd.DataFrame(np.ones([self.tn,6]),index=self.node_list,columns=["x","y","z","thetax","thetay","thetaz"])
                boundarycondition.sort_index(inplace=True)
                self.boundarycondition.loc[boundarycondition.index]= boundarycondition.loc[:]
        else:
            raise Exception ("These nodes in boundary condition does not exist: ",  [i for i, val in enumerate(self.__bc_index) if not val] )


        self.autoflooring= autoflooring
        self.self_weight= self_weight
        self.slab_details= slab_details


        if load_combo is None:
            raise Exception ("Use 'Strucpy.RCF' for None load combination")
        if load_combo is not None:
            if not isinstance(load_combo, pd.DataFrame):
                raise TypeError ("Type of the 'load_combo' must be of  type 'DataFrame'")
   
            self.load_combo= load_combo
            self.load_combo.columns= ["Dead_Load","Live_Load","EQX","-EQx","EQZ","-EQZ"]
            self.load_combo.fillna(0,inplace=True)

            self.load_combo.index= [i for i in range (1, len(self.load_combo)+1)]


        self.seismic_def= seismic_def
        self.point_loads = None

        self.infillwall= infillwall
        self.grade_conc= grade_conc
        self.col_stablity_index= col_stablity_index

        E0= 5000*np.sqrt(self.grade_conc)*(10**3)
        alpha0= 10*(10**(-6))
        mu0= 0.17           
        G0= E0/(2*(1+mu0))
        self.__concrete_density= 25

        self.__DefaultMproperties= pd.DataFrame({ "Type": "All", "Material": "Concrete", "Grade M-": self.grade_conc, "Density (kN/m3)": self.__concrete_density, "Young Modulus (kN/m2)": E0, "Poisson's Ratio (mu)": mu0, "Thermal Coefficient (alpha)": alpha0, "Critical Damping": 0.05, "Modulus of Rigidity (kN/m2)": G0}, index=[1])

        if properties is None:
            self.Mproperties= self.__DefaultMproperties.copy()

        if properties is not None:
            if not isinstance(properties, pd.DataFrame):
                raise TypeError ("Type of the 'Material Properties' must be of  type 'DataFrame'")

            typelist = ['all', 'beam', 'column', 'slab']

            properties.columns= ["Type", "Material", "Grade M-", "Density (kN/m3)", "Young Modulus (kN/m2)", "Poisson's Ratio (mu)", "Thermal Coefficient (alpha)", "Critical Damping", "Modulus of Rigidity (kN/m2)"]

            for kk in range (len(properties.index)):
                if not isinstance(properties.iat[kk,0], str):
                    raise Exception ("Something wrong with the 'Type' in Material Properties. It can only be 'All', 'Beam', 'Column' or 'Slab'. ")
            
            if (properties['Type'].str.lower().isin(typelist)).all()==True:
                TypeM= properties['Type'].to_list()
            else:
                raise Exception ("Something wrong with the 'Type' in Material Properties. It can be only 'All', 'Beam', 'Column' or 'Slab'")
            
            for kk in range (len(properties.index)):
                if not isinstance(properties.iat[kk,1], str):
                    raise Exception ("Something wrong with the name of the material in Material Properties")
            NameM= properties['Material'].to_list()

            for kk in range (len(properties.index)):
                if isinstance(properties.iat[kk,2].item(), (float, int)):
                    pass
                else:
                    raise Exception ("Something wrong with the grade of concrete in Material Properties- It must be Int or float number like 20, 25 ,30, 37.5 etc. representing the M20, M25, M30 respectively. ")

            for kk in range (len(properties.index)):
                if properties.iloc[kk,2] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the grade of concrete in second column of Material Properties. It must be positive number.")
            gradeM= properties['Grade M-'].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,3] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of density in second column of Material Properties. It must be positive number.")
            densityM= properties['Density (kN/m3)'].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,4] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Young Modulus in third column of Material Properties. It must be positive number.")
            EM= properties['Young Modulus (kN/m2)'].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,5] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Poisson's Ratio in fourth column of Material Properties. It must be positive number.") 
            muM= properties["Poisson's Ratio (mu)"].to_list()

            for kk in range (len(properties.index)):                 
                if properties.iloc[kk,6] != 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Thermal Coefficient in fifth column of Material Properties") 
            alphaM= properties["Thermal Coefficient (alpha)"].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,7] > 0:
                    pass
                elif properties.iloc[kk,7] == 0:
                    properties.iloc[kk,7]= 0.05
                else:
                    raise Exception ("Something wrong with the value of Critical Damping in sixth column of Material Properties") 
            critM= properties["Critical Damping"].to_list()

            for kk in range (len(properties.index)):
                if properties.iloc[kk,8] > 0:
                    pass
                else:
                    raise Exception ("Something wrong with the value of Modulus of Rigidity in seventh column of Material Properties. It must be positive number.")
            GM= properties["Modulus of Rigidity (kN/m2)"].to_list()

            prop_index= [i for i in range (1,len(properties)+1)]
            
            self.Mproperties= pd.DataFrame({ "Type": TypeM ,"Material": NameM, "Grade M-": gradeM, "Density (kN/m3)": densityM, "Young Modulus (kN/m2)": EM, "Poisson's Ratio (mu)": muM, "Thermal Coefficient (alpha)": alphaM, "Critical Damping": critM, "Modulus of Rigidity (kN/m2)": GM}, index= prop_index)


        self.__OB=  [_RCFforenvelop.remote(self.nodes_details, self.member_details, self.boundarycondition, forcesnodal= self.forcesnodal, autoflooring= self.autoflooring, slab_details= self.slab_details, seismic_def= self.seismic_def, self_weight= self.self_weight, infillwall= self.infillwall, properties= self.Mproperties, col_stablity_index= self.col_stablity_index, load_combo= self.load_combo.iloc[[i]] ) for i in range (len(load_combo)) ]


        self.LClist= [f"LC{i}" for i in(self.load_combo.index)]
            
        self.__ENVanalysis_react= False
        self.__ENVanalysis_dis = False
        self.__ENVanalysis_memF = False
        self.__ENVanalysis_def = False

        self.__Anlaysisperformed= False
        self.__PreP_status= False

        self.__mdd= self.member_details.copy()
        self.__ndd= self.nodes_details.copy()
        self._bcd= self.boundarycondition.copy()

    def preP(self):
        """This function of class :class:`StrucPy.RCFA.RCFenv` performs pre processing for the analysis of a Reinforced Concrete Frame that user intend to analyse for different load combinations. It generates all the pre analysis data. This function should be called before performing analysis.

        :param: None
        :return: None
        """
        self.__PreP_status= True

        [self.__OB[i].preP.remote() for i in range (len(self.__OB))]

    def RCanalysis(self):
        """This function of class :class:`StrucPy.RCFA.RCFenv` 
        performs analysis for a Reinforced Concrete Frame that user intend to analyse for different load combinations. It generates all the post analysis data. This function should be called first before any other function.

        :param: None
        :return: None
        """
        if self.__PreP_status== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        [self.__OB[j].RCanalysis.remote() for j in range (len(self.__OB))]

        self.memD,self.slab_pd, self.baseN, self.seismic_def, self.M_prop = ray.get(self.__OB[4].getGlobalVariables.remote())

        self.__Anlaysisperformed= True


    def getTLC(self):
        """Returns a *List* of *ray.actors* of class :class:`StrucPy.RCFA.RCFenv` presenting the different ray actors capable of creating and executing class :class:`StrucPy.RCFA.RCF` for different load combination for the analysis of reinforced concrete frame as passed by user. It can be used to retrieve further data from class :class:`StrucPy.RCFA.RCF`


        :param: None
        :return: A list of objects representing different Load Combination objects
        :rtype: List
        """
        return(self.__OB)
    
    def __evalENVreact(self):

        self.__ENVanalysis_react= True

        self.__reactionsENV= pd.DataFrame()
        self.__dsgReaction= pd.DataFrame()
        col_ind3= ["Max Reaction", "Load Combination"]

        for i in (self.baseN.index):
            react_node= pd.DataFrame()
            col_ind1=[] 
            col_ind2=[]
            s=1
            for j in self.__OB:
                tt= ray.get(j.reactions.remote())
                tt= tt[tt.index==i]
                react_node= pd.concat([react_node,tt],ignore_index=True)
                col_ind1.append(i)
                col_ind2.append(f"LC{s}")
                s=s+1
                index_new =[
                    col_ind1,
                    col_ind2]
            
            dsg_index= [
                    [i,i],
                    col_ind3]
            react_node.set_index(index_new, inplace=True)
            dsgReact= (react_node.loc[:].max(numeric_only= True)).to_frame().T
            dsgReact_index= react_node.loc[i].idxmax().to_frame().T
            
            for k in range (6):
                if react_node.iat[0,k]<0:
                    dsgReact.iat[0,k]= react_node.iloc[:,k].min()
                    dsgReact_index.iat[0,k]= react_node.loc[i].idxmin().to_list()[k]

            
            dsgReact= pd.concat([dsgReact,dsgReact_index],ignore_index= True)
            dsgReact.set_index(dsg_index, inplace=True)
            


            self.__dsgReaction= pd.concat([self.__dsgReaction,dsgReact])
            self.__reactionsENV=pd.concat([self.__reactionsENV,react_node])
        
    def __evalENVGdisp(self):

        self.__ENVanalysis_dis = True

        self.__GdispENV= pd.DataFrame()
        self.__dsgGdisp= pd.DataFrame()
        self.memDENV= pd.DataFrame()
        self.dispLENV= pd.DataFrame()
        self.dispGENV= pd.DataFrame()
        self.driftENV= pd.DataFrame()
        self.seismicshearENV= pd.DataFrame()
        nodes= list(set(self.nodes_details.index) - set(self.baseN.index))
        col_ind3= ["Max. Displacement", "Load Combination"]
        for i in (nodes):
            Disp_node= pd.DataFrame()
            col_ind1=[] 
            col_ind2=[]

            s=1
            for j in self.__OB:
                tt= ray.get(j.Gdisp.remote())
                tt= tt[tt.index==i]
                Disp_node= pd.concat([Disp_node,tt],ignore_index=True)
                col_ind1.append(i)
                col_ind2.append(f"LC{s}")
                s=s+1
                index_new =[
                    col_ind1,
                    col_ind2]
            
            dsg_index= [
                    [i,i],
                    col_ind3]
            Disp_node.set_index(index_new, inplace=True)
            dsgdisp= (Disp_node.loc[:].max(numeric_only= True)).to_frame().T
            dsgdisp_index= Disp_node.loc[i].idxmax().to_frame().T
            
            for k in range (6):
                if Disp_node.iat[0,k]<0:
                    dsgdisp.iat[0,k]= Disp_node.iloc[:,k].min()
                    dsgdisp_index.iat[0,k]= Disp_node.loc[i].idxmin().to_list()[k]
            
            dsgdisp= pd.concat([dsgdisp,dsgdisp_index],ignore_index= True)
            dsgdisp.set_index(dsg_index, inplace=True)

            self.__dsgGdisp= pd.concat([self.__dsgGdisp,dsgdisp])
            self.__GdispENV=pd.concat([self.__GdispENV,Disp_node])

    def __evalENVmemF(self):
        
        self.__ENVanalysis_memF= True

        self.__memendForces= pd.DataFrame()
        self.__memmaxForces= pd.DataFrame()
        self.__dsgmaxForces= pd.DataFrame()

        col_ind4= ["Fx (kN)", "Fy (kN)", "Fz (kN)", "Mx (kN-m)", "My (kN-m)", "Mz (kN-m)"]
        tm= self.tm
        col_ind3= ["+ve", "-ve"]

        for i in range (tm):
            s=1
            for j in self.__OB:
                col_ind1=[] 
                col_ind2=[]
                tt= ray.get(j.memF.remote())[i]
                po1= self.memD.iat[i,0]
                po2= self.memD.iat[i,1]
                pos= [po1, po2]
                endforces= np.empty([2,6])
                endforces[0,:]= tt[0, 1:]
                endforces[1,:]= tt[-1, 1:]

                maxforces= np.empty([2,6])
                maxforces[0,:]=np.max(tt,axis=0)[1:]  #positive Forces
                maxforces[1,:]= np.min(tt,axis=0)[1:]      #Negative Forces
                            
                col_ind1.append(self.member_list[i])
                col_ind2.append(f"LC{s}")

                col_ind5= col_ind1

                index_max= [
                    col_ind1,
                    col_ind2,
                    col_ind3]


                col_ind1.append(self.member_list[i])
                col_ind2.append(f"LC{s}")

                s=s+1
                index_new =[
                    col_ind1,
                    col_ind2]          
                



                lcmaxF= pd.DataFrame(maxforces, index=index_max,columns=col_ind4 )

                self.__memmaxForces= pd.concat([self.__memmaxForces,lcmaxF])


                memENDF=pd.DataFrame(endforces, index= index_new ,columns=col_ind4)
                memENDF.insert(0,"Node",pos)
                self.__memendForces= pd.concat([self.__memendForces,memENDF])

            index_max_dsg =[
                    col_ind5,
                    ["Design Forces", "Load Combination"]]
            
            dsgFposi= self.__memmaxForces.max().to_frame().T
            dsgFneg= self.__memmaxForces.min().to_frame().T
            dsgFposi_index= self.__memmaxForces.idxmax().to_list()
            dsgFneg_index= self.__memmaxForces.idxmin().to_list()

            dsgF= pd.DataFrame(index=index_max_dsg, columns=col_ind4 )
            
            for i in range (6):
                if abs(dsgFposi.iat[0,i]) > abs(dsgFneg.iat[0,i]):
                    dsgF.iat[0,i]= dsgFposi.iat[0,i]
                    dsgF.iat[1,i]= dsgFposi_index[i][1] 
                else:
                    dsgF.iat[0,i]= dsgFneg.iat[0,i]
                    dsgF.iat[1,i]= dsgFneg_index[i][1]
            
            self.__dsgmaxForces= pd.concat([self.__dsgmaxForces,dsgF ])

        self.__memmaxForces.loc[:,:,"+ve"]= self.__memmaxForces.loc[:,:,"+ve"].where(self.__memmaxForces.loc[:,:,"+ve"]>=0, "NA")

        self.__memmaxForces.loc[:,:,"-ve"]= self.__memmaxForces.loc[:,:,"-ve"].where(self.__memmaxForces.loc[:,:,"-ve"]<0, "NA")

        # MAx value need to be calculated for design

    def __evalENVdef(self):
        
        self.__ENVanalysis_def= True

        self.__memlocdis= pd.DataFrame()
        self.__memglodis= pd.DataFrame()

        self.__dsgDefL= pd.DataFrame()
        self.__dsgDefG= pd.DataFrame()

        col_ind4= ["Defx (mm)", "Defy (mm)", "Defz (mm)"]
        tm= self.tm
        col_ind3= ["Maximum Displacement", "Load Combination"]

        for i in range (tm):
            s=1
            dsgLmax=  np.zeros((1,3))
            dsgGmax=  np.zeros((1,3))

            dsg_indexL= []
            dsg_indexG= []

            pos= self.member_list[i]

            for j in self.__OB:
                col_ind1=[] 
                col_ind2=[]
                localdef= np.zeros((1,3))
                globaldef= np.zeros((1,3))

                col_ind1.append(pos)
                col_ind2.append(f"LC{s}")

                s=s+1
                index_new =[
                    col_ind1,
                    col_ind2]

                ld= ray.get(j.defLD.remote()) [i][:,1:]
                gd= ray.get(j.defGD.remote()) [i][:,1:]

                defLmax=np.amax(ld,axis=0)
                defLmin=np.amin(ld,axis=0)

                defGmax=np.amax(gd,axis=0)
                defGmin=np.amin(gd,axis=0)
 
                for k in range (3):
                    if abs(defLmax[k]) >= abs(defLmin[k]):
                        localdef[0,k]= defLmax[k]
                    else:
                        localdef[0,k]= defLmin[k]

                    if abs(defGmax[k]) >= abs(defGmin[k]):
                        globaldef[0,k]= defGmax[k]
                    else:
                        globaldef[0,k]= defGmin[k]    

            
    
                Ldef= pd.DataFrame(localdef, index= index_new, columns=col_ind4 )
                Gdef= pd.DataFrame(globaldef, index= index_new, columns=col_ind4 )

                self.__memlocdis= pd.concat([self.__memlocdis,Ldef])
                self.__memglodis= pd.concat([self.__memglodis,Gdef])


            
            Ldef1= self.__memlocdis.loc[pos].copy()
            Gdef1= self.__memglodis.loc[pos].copy()

            LdefIndex= Ldef1.index.to_list()
            GdefIndex= Gdef1.index.to_list()
            
            index_dsg= [
                        [pos,pos], 
                        col_ind3 ]
            for l in range (3):
                if  abs(Ldef1.iat[0,l]) >= abs (Ldef1.iat[1,l]):
                    dsgLmax[0,l]= Ldef1.iat[0,l]
                    dsg_indexL.append(LdefIndex[0])
                else:
                    dsgLmax[0,l]= Ldef1.iat[1,l]
                    dsg_indexL.append(LdefIndex[1])
                
                if  abs(Gdef1.iat[0,l]) >= abs (Gdef1.iat[1,l]):
                    dsgGmax[0,l]= Gdef1.iat[0,l]
                    dsg_indexG.append(GdefIndex[0])
                else:
                    dsgGmax[0,l]= Gdef1.iat[1,l]
                    dsg_indexG.append(GdefIndex[1])

            dsgLdef= pd.DataFrame(dsgLmax,  columns=col_ind4 )

            dsgGdef= pd.DataFrame(dsgGmax, columns=col_ind4 )
            
            dsgLdef.loc[len(dsgLdef.index)] = dsg_indexL 
            dsgGdef.loc[len(dsgGdef.index)] = dsg_indexG 
            
            dsgLdef.set_index(index_dsg, inplace=True)
            dsgGdef.set_index(index_dsg, inplace=True)

            self.__dsgDefL= pd.concat([self.__dsgDefL,dsgLdef])
            self.__dsgDefG= pd.concat([self.__dsgDefG,dsgGdef])


    def changeFL(self, floor=None,thickness= None, LL=None, FF=None, WP=None, delf=None ):
        """This non returing function of :class:`StrucPy.RCFA.RCFenv` objects performs change in floor/slab. If 'floor' argument is none changes are made to every floor/slab and if floor number is passed in argument 'floor', changes will be made in that particular floor/slab.

        1. Changes can be performed on all floor together.
        2. Changes can be performed on only selected floor together.   
        3. Thickness of floors/slabs can be changed.
        4. Live Load on floors/slabs can be changed.
        5. Floor Finish Load on floors/slabs can be changed.
        6. Water Proffing Load on floors/slabs can be changed.
        7. Any particular floor or the group of floor can be deleted. 
        
        :param floor: Optional, Default as None (Changes to all floors/slabs).  It represents floor numbers/names on which chages has to be made. If changes has to be made on particular floor/slab or a group of floor/slab, number/name of the floors/slab has to be passed in argument 'floor'. 
        :type floor: int, list

        :param thickness: Optional, Default as None (No changes in thickness of floors/slabs).  It represents thickness of floor/slab in "millimeter(mm)".  
        :type thickness: int, list

        :param LL: Optional, Default as None (No changes in live load acting on the floors/slabs).  It represents represents the live load acting on the floors/slabs in "kiloNewton/square meter (kN/m2)".  
        :type LL: int, list

        :param FF: Optional, Default as None (No changes in floor finishing load acting on the floors/slabs).  It represents represents the floor finish load acting on the floors/slabs in "kiloNewton/square meter (kN/m2)".  
        :type FF: int, list 

        :param WP: Optional, Default as None (No changes in water proofing load acting on the floors/slabs).  It represents represents the water proofing load acting on the floors/slabs in "kiloNewton/square meter (kN/m2)".  
        :type WR: int, list  

        :param delf: Optional, Default as None (No deletion of any floor/slab).  It represents floor numbers/names which has to be deleted. If particular floor/slab or a group of floor/slab, number/name of the floors/slab has to be deleted, number/names of the floor has to passed in argument 'delf'.  
        :type delf: int, list   
        :return: None  
        """

        if self.__PreP_status == False:
            raise Exception("Perform Pre-Processing of the structure using method 'preP' before changing floor details")

        [self.__OB[i].changeFL.remote(floor,thickness, LL, FF, WP, delf ) for i in range (len(self.__OB))]

        self.__Anlaysisperformed= False
        self.__ENVanalysis_react= False
        self.__ENVanalysis_dis= False
        self.__ENVanalysis_memF= False
        self.__ENVanalysis_def= False


    def getReact(self):

        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects. It presents the reactions of reinforced concrete frame from every load combinations.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """ 
        if self.__ENVanalysis_react== True:
            pass
        else:
            self.__evalENVreact()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")
        
        return (self.__reactionsENV)

    def getReactmax(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects. It presents the 'maximum' reactions of reinforced concrete frame (among every load combinations) for the final analysis and design.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """
        if self.__ENVanalysis_react== True:
            pass
        else:
            self.__evalENVreact()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__dsgReaction)
    
    def getNdis(self):

        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects. It presents the nodal displacement of reinforced concrete frame for every load combinations.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """
        if self.__ENVanalysis_dis== True:
            pass
        else:
            self.__evalENVGdisp()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__GdispENV)

    def getNdismax(self):
        
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects. It presents the maximum nodal displacement of reinforced concrete frame (among every load combinations) for the final analysis and design.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """
        if self.__ENVanalysis_dis== True:
            pass
        else:
            self.__evalENVGdisp()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__dsgGdisp)


    def getEndMF(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects. It presents the 'End Forces' for every member of reinforced concrete frame for every load combinations in local coordinate of a member.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """

        if self.__ENVanalysis_memF== True:
            pass
        else:
            self.__evalENVmemF()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__memendForces)

    def getMFmax(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects. It presents the 'Maximum positive and Negative Member Forces' for every member of reinforced concrete frame from every load combinations in local coordinate system of a member.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """
        if self.__ENVanalysis_memF== True:
            pass
        else:
            self.__evalENVmemF()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__memmaxForces)

    def getMFdsg(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects. It presents the 'Maximum Member Forces' for every member of reinforced concrete frame in local coordinate system of a member. It can be used for the final analysis and design.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """
        if self.__ENVanalysis_memF== True:
            pass
        else:
            self.__evalENVmemF()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__dsgmaxForces)


    def getLDef(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects presenting maximum deflection detail of members in reinforced concrete frame from every load combinations in 'Local Coordinate System'. 
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """ 
        if self.__ENVanalysis_def== True:
            pass
        else:
            self.__evalENVdef()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__memlocdis)

    def getGDef(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects presenting maximum deflection detail of members in reinforced concrete frame from every load combinations in 'Global Coordinate System'. 
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """

        if self.__ENVanalysis_def== True:
            pass
        else:
            self.__evalENVdef()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__memglodis)

    def getLDefmax(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects presenting maximum deflection detail of members in reinforced concrete frame (among every load combinations) in 'Local Coordinate System'. It can be used for the final result analysis and design.
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """
        if self.__ENVanalysis_def== True:
            pass
        else:
            self.__evalENVdef()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__dsgDefL)

    def getGDefmax(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects presenting maximum deflection detail of members in reinforced concrete frame (among every load combinations) in 'Gocal Coordinate System'. It can be used for the final result analysis and design. 
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """ 
        if self.__ENVanalysis_def== True:
            pass
        else:
            self.__evalENVdef()

        if self.__Anlaysisperformed== False:
            raise Exception ("Perform analysis of the structure first.")

        return (self.__dsgDefG)
    
    def getLClist(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCFenv` objects presenting maximum deflection detail of members in reinforced concrete frame (among every load combinations) in 'Gocal Coordinate System'. It can be used for the final result analysis and design. 
         
        :param: None
        :return: A multiindex DataFrame of :class:`StrucPy.RCFA.RCFenv` objects.
        :rtype: DataFrame
        """ 
        LC= self.load_combo.copy()
        LC.insert(0, "Name", self.LClist)
        
        return (LC)
    
    def changeFrame(self, member, node= None):
        """This non returing function of :class:`StrucPy.RCFA.RCF` object performs changes by deleting members and nodes of Reinforced Concrete Frame. It changes the frame model. 
        :ref:`InputExample:changeFrame` for more details.

        :param member: Member ID/ID's which has be deleted from the frame.
        :type member: int/list

        :param node: node ID/ID's which has be deleted from the frame.
        :type node: int/list
        :return: None
        """

        if not isinstance(member, (int, list)):
            raise Exception ("The member ID provided is of wrong type. It can only be int,float or list ")
        
        if isinstance(member, int):
            member_nodes_check= self.__member_details.index.isin([member])

            if member_nodes_check.any():
                self.__member_details.drop([member], inplace= True)
            else:
                raise Exception (f"These {member} member ID does not exist. " )
            
        if isinstance(member, list):

            member_nodes_check=    pd.Series(member).isin(self.__member_details.index)

            if member_nodes_check.all():
                self.__member_details.drop(member, inplace= True)

            else:
                raise Exception ("These nodes does not exist in member_details: ",  [i for i, val in enumerate(member_nodes_check) if not val] )
          


        if node is not None:
            if not isinstance(node, (int, list)):
                raise Exception ("The nodes ID provided is of wrong type. It can only be int,float or list ")

        if isinstance(node, int):
            nodes_check= self.__nodes_details.index.isin([node])

            if nodes_check.any():
                member_nodes_check= self.__member_details.loc[:,["Node1", "Node2"]].isin([node])

                if member_nodes_check.any().any():
                    raise Exception ("These node can not be deleted as it is being used by a member. First delete the member in order to remove the nodes")
                
                self.__nodes_details.drop([node], inplace= True)
            else:
                raise Exception (f"The {node} node ID does not exist. " )
            
        if isinstance(member, list):
            nodes_check=    pd.Series(node).isin(self.__nodes_details.index)

            if nodes_check.all():
                member_nodes_check= self.__member_details.loc[:,["Node1", "Node2"]].isin(node)

                if member_nodes_check.any().any():
                    raise Exception ("These node can not be deleted as it is being used by a member. First delete the member in order to remove the nodes")

                self.__nodes_details.drop(node, inplace= True)
            else:
                raise Exception ("These nodes does not exist in details: ",  [i for i, val in enumerate(nodes_check) if not val] ) 
            
        n1= self.__member_details.iloc[:,0:2].to_numpy()
        orphan_nodes_status = self.__nodes_details.index.isin(n1.flatten())
        
        orphan_nodes= [i for i, val in enumerate(orphan_nodes_status) if not val]

        self.__nodes_details.drop(orphan_nodes, inplace= True)

        self.__PreP_status= False

    def changeBoundcond(self, bound_conditions):
        """This non returing function of :class:`StrucPy.RCFA.RCF` object performs changes in boundary condition of Reinforced Concrete Frame. It completely  replaces the boundary existing boundary consition with the passed one. :ref:`InputExample:changeBoundcond` for more details.

        :param bound_conditions: Details of the new boundary condition to be used in analysis.
        :type bound_conditions: Dataframe
        :return: None
        """

        if not isinstance(bound_conditions, pd.DataFrame):
            raise Exception ("The bound_conditions provided is of wrong type. It can only be dataframe ")
        
        if len(bound_conditions.columns) != 6:
            raise Exception ("The boundary condition dataframe must contain 6 columns representing each degree of freedom in 3D space i.e. 'Trans x', 'Trans y', 'Trans z', 'Rotation x', 'Rotation y', 'Rotation z'. ")
        
        self.__bc_index = bound_conditions.index.isin(self.__nodes_details.index)

        if self.__bc_index.all():
            if len(bound_conditions)==self.node_list:
                self.__boundarycondition= bound_conditions.sort_index()
                self.__boundarycondition.columns= ["x","y","z","thetax","thetay","thetaz"]
            elif len(bound_conditions)!=self.node_list:             
                self.__boundarycondition = pd.DataFrame(np.ones([self.tn,6]),index=self.node_list,columns=["x","y","z","thetax","thetay","thetaz"])
                bound_conditions.sort_index(inplace=True)
                self.__boundarycondition.loc[bound_conditions.index]= bound_conditions.loc[:]
        else:
            raise Exception ("These nodes in boundary condition does not exist: ",  [i for i, val in enumerate(self.__bc_index) if not val] )

        self.__PreP_status= False

    def modelND(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the nodes of the reinforced concrete frame. It a parameter passed by user or generated by agrumunet `framegen`.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  
        return (self.__ndd)
    
    def modelMD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the members of the reinforced concrete frame. It a parameter passed by user or generated by agrumunet `framegen`.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  

        return (self.__mdd)
    
    def modelBCD(self):
        """Returns a *DataFrame* of :class:`StrucPy.RCFA.RCF` object presenting the boundary condition of the reinforced concrete frame. It a parameter passed by user or generated by agrumunet `framegen`.
         
        :param: None
        :return: A *DataFrame* of :class:`StrucPy.RCFA.RCF` objects.
        :rtype: *DataFrame* 
        """  

        return (self._bcd)

