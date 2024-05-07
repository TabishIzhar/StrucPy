import numpy as np
import pandas as pd
import plotly.graph_objects as go 
import itertools as itr  
from plotly.subplots import make_subplots     
import copy
import time
import ray
from ._arrangeCal import *

@ray.remote
class _RCFforenvelop():
        
    def __init__(self, nodes_details, member_details, boundarycondition, forcesnodal=None, slab_details=None, load_combo=None, seismic_def=None,self_weight= True, infillwall=False, autoflooring= False, properties= None, col_stablity_index= 0.04):
        
        
        self.__nodes_details= nodes_details.copy()
        self.__member_details= member_details.copy()

        self.member_list= self.__member_details.index.to_list()
        self.node_list= self.__nodes_details.index.to_list()
        self.tn= self.__nodes_details.shape[0]
        self.tm= self.__member_details.shape[0]

        
        self.__forcesnodal = forcesnodal.copy()
        self.__boundarycondition= boundarycondition.copy()


        self.autoflooring= autoflooring
        self.__self_weight= self_weight


        if slab_details is None and self.autoflooring== False:
            self.__slabload_there= 0
            self.__slab_details= "Slab/Floor not present in the Frame"
        if slab_details is not None and self.autoflooring== False:
            if not isinstance(slab_details, pd.DataFrame):
                raise TypeError ("Type of the 'slab_details' must be DataFrame")
            self.__slabload_there= 1
            self.__slab_details= slab_details.sort_index().copy()
            self.__slab_details.fillna(0,inplace=True)
            self.__slab_details.columns = ['Node1', 'Node2', 'Node3', 'Node4',	'Thickness(mm)', 'FF(kN/m2)', 'LL(kN/m2)', 'Waterproofing(kN/m2)']

        if slab_details is None and self.autoflooring== True:
            self.__slabload_there= 1


        self.load_combo= load_combo.copy()

        if seismic_def is None:
            self.__seismic_def_status= False
            self.__seismic_def= "No Seismic Anlaysis Performed"
        else:
            self.__seismic_def_status= True
            if not isinstance(seismic_def,pd.DataFrame): 
                raise TypeError ("Type of the 'seismic_def' must be DataFrame")
            self.__seismic_def= seismic_def.copy()
            self.__seismic_def.columns= ["Z","I","R","Sag","Damping(%)","Soil Type", "Time Period" ]



        self.__Mproperties= properties.copy()
        self.__col_stablity_index= col_stablity_index


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
        self.__floor_loads= None
        self.__beam_loads= None
        self.__column_loads= None
        self.story_lumploads= None
        self.__infillwall= infillwall
        
        self.__SeismicShear= None
        self.__floor_loads= None

        self.baseN= None
        self.__bd_LDeduct= None
        self.__ds= []
        self.__Analysis_performed= False
        self.__PreP_status= False


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
        
        self.__mdd= self.__member_details.copy()
        self.__ndd= self.__nodes_details.copy()
        self._bcd= self.__boundarycondition.copy()


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
        self.__nodes_detail_preP= self.nodes_detail.copy()


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

        self.__beams_detail_preP= self.beams_detail.copy()     
        self.__columns_detail_preP= self.columns_detail.copy()
        self.__nodes_detail_preP= self.nodes_detail.copy() 


    def __autoflooring(self):  

        self.__autoflooring_done = True
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
        
                self.floor_loading = Loadings_floor

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
            self.column_Lumploads = self.__column_Lumploads
            self.beam_Lumploads = self.__beam_Lumploads
             
        else:
            TSL_self_weight= Loadings_members.loc[:,'Self-Weight(kN)']
            TSL_other_dead_load= Loadings_members.loc[:,'Other-Dead-Loads']

            self.story_lumploads = pd.concat([TSL_self_weight,TSL_other_dead_load,story_ht_pd],axis=1)
            self.column_Lumploads = self.__column_Lumploads
            self.beam_Lumploads = self.__beam_Lumploads

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
        self.len_beam=[]
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

        self.__ds= []
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

        even_index= [i for i in range (0,len(maxforces_pd), 2)]
        odd_index= [i for i in range (1,len(maxforces_pd), 2)]

        maxforces_pd[maxforces_pd.iloc[even_index,:]<0]= 9999999999
        maxforces_pd[maxforces_pd.iloc[odd_index,:]>0]= 9999999999

        maxforces_pd.replace(9999999999, "--", inplace=True)

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
        else:
            T = self.__seismic_def['Time Period'].item()

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
        else:
            Sag = self.__seismic_def['Sag'].item()

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
        self.__nodal_S_F= pd.DataFrame()

        for i in range (1,len(ND.Floor.unique())):
            ND_f=  ND.loc[ND['Floor']==i]

            Stiff_ratio= ND_f['Stiffness']/ (ND_f['Stiffness'].sum())

            nodal_seismic_forces= Vi[i]*Stiff_ratio

            nodal_seismic_forces_pd=  nodal_seismic_forces.to_frame()
            nodal_S_F= pd.DataFrame( nodal_seismic_forces_pd.to_numpy(), index= nodal_seismic_forces_pd.index, columns= ["Nodal Forces"] )

            self.__nodal_S_F= pd.concat([self.__nodal_S_F,nodal_S_F])

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
        if self.__PreP_status== True:
            pass

        self.__PreP_status = True

        self.__nodes_arrangement_for_members()

        if self.tm < 150:
            self.__arrange_beam_column_nodes()

        if self.tm > 150:
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
        if self.__PreP_status == False:
            raise Exception("Perform Pre Processing of the structure using method 'preP'")

        self.__Analysis_performed= True

        self.__properties()
        self.__member_detailing()


        if self.__slabload_there==1:
            self.__floorLoading()
        self.__tloads()
       
        self.__stiffnessbeam()

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


        self.__solution()

        self.__internal_forces_calculation()

        self.__CalDeflections()

        self.__calMaxmemF()

        if self.__seismic_def_status == True:
            self.__drift()

        self.__effectiveLength()

    def model3D(self):

        if self.__PreP_status!= True:
            raise Exception ("Preform Pre-processing first")


        xx= self.__nodes_details.to_numpy()
        nodetext= self.__nodes_details.index.to_numpy()
        xxx= self.__cords_member_order.to_numpy()
        tmm=len(xxx)
        fig1= go.Figure()
        fig2= go.Figure()
        fig3= go.Figure()
        fig4= go.Figure()
        fig5= go.Figure()

        fig1.add_trace(go.Scatter3d(x=xx[:,2],y=xx[:,0],z=xx[:,1],mode='markers+text', text=nodetext,textposition="middle right"))
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
                        width=3),name= f"member {kk+1}" ))

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

        Model.update_layout(height=1000, width=1600)

        # fw= go.FigureWidget(Model)
        return (Model)



    def sfbmd(self, element):

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

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[sfy[0], sfy[-1]],
            mode="markers+text",
            text=[sfy[0], sfy[-1]],
            textposition=["top center", "bottom center"],
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=1)


        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=1, col=2)
        fig.add_trace(go.Scatter(x=[0,0], y=[0,sfz[0]],mode='lines', line=dict(color="red") ), row=1, col=2)
        fig.add_trace(go.Scatter(x=xx, y=sfz,mode='lines',line=dict(color="red")), row=1, col=2)
        fig.add_trace(go.Scatter(x=[xx[-1],xx[-1]], y=[0,sfz[-1]],mode='lines', line=dict(color="red") ), row=1, col=2)
             
        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[sfz[0], sfz[-1]],
            mode="markers+text",
            text=[sfz[0], sfz[-1]],
            textposition=["top center", "bottom center"],
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=2)

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=2, col=1)
        fig.add_trace(go.Scatter(x=[0,0], y=[0,bmy[0]],mode='lines', line=dict(color="red") ), row=2, col=1)
        fig.add_trace(go.Scatter(x=xx, y=bmy,mode='lines',line=dict(color="red")), row=2, col=1)
        fig.add_trace(go.Scatter(x=[xx[-1],xx[-1]], y=[0,bmy[-1]],mode='lines', line=dict(color="red") ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[bmy[0], bmy[-1]],
            mode="markers+text",
            text=[bmy[0], bmy[-1]],
            textposition=["top center", "top center"],
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=1)

        fig.add_trace(go.Scatter(x=xx, y=yyb,mode='lines', line=dict(color="#000000") ),row=2, col=2)
        fig.add_trace(go.Scatter(x=[0,0], y=[0,bmz[0]],mode='lines', line=dict(color="red") ), row=2, col=2)
        fig.add_trace(go.Scatter(x=xx, y=bmz,mode='lines',line=dict(color="red")), row=2, col=2)
        fig.add_trace(go.Scatter(x=[xx[-1],xx[-1]], y=[0,bmz[-1]],mode='lines', line=dict(color="red") ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[bmz[0], bmz[-1]],
            mode="markers+text",
            text=[bmz[0], bmz[-1]],
            textposition=["top center", "top center"],
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=2)

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

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[defx[0], defx[-1]],
            mode="markers+text",
            text=[defx[0], defx[-1]],
            textposition="top center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[defy[0], defy[-1]],
            mode="markers+text",
            text=[defy[0], defy[-1]],
            textposition="top center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[defz[0], defz[-1]],
            mode="markers+text",
            text=[defz[0], defz[-1]],
            textposition="top center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=3, col=1)


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

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[defx[0], defx[-1]],
            mode="markers+text",
            text=[defx[0], defx[-1]],
            textposition="top center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[defy[0], defy[-1]],
            mode="markers+text",
            text=[defy[0], defy[-1]],
            textposition="top center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xx[-1]],
            y=[defz[0], defz[-1]],
            mode="markers+text",
            text=[defz[0], defz[-1]],
            textposition="top center",
            textfont=dict(
            size=12,
            color="blue")
            ),row=3, col=1)

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
                        width=1)))
        
        Model.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   yaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   zaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
             ))

        for i in range(self.tm):
            Model.add_trace(go.Scatter3d(x=xxx1[i][:,2],y=xxx1[i][:,0],z=xxx1[i][:,1], mode='lines',      
                line=dict(
                        color="red",                # set color to an array/list of desired values
                        width=1)))



        Model.update_layout(height=800, width=800, title_text=f"Deflection of Structure")
        return(Model)

    def reactions(self):      
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        GForces= self.__GForces.copy()
        GForces= GForces /1000       #converting into kN from N
        GForces = np.round(GForces.astype(np.double),3)
        Forcepd = pd.DataFrame(GForces, columns = ['Fx (kN)','Fy (kN)','Fz (kN)','Mx (kN-m)','My (kN-m)','Mz (kN-m)'], index = self.__nodes_details.index.to_list())  

        react= Forcepd.loc[self.baseN.index] 
        return react.sort_index()

    def Gdisp(self):        # Returns Diaplacement GLOBAL
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        dispmat1= self.__GDisplacement.copy()
        dispmat1[:,0:3] = (dispmat1[:,0:3])*1000    # converting into mm
        dispmat1 = np.round(dispmat1.astype(np.double),3)
        Disppd = pd.DataFrame(dispmat1, columns = ['ux (mm)','uy (mm)','uz (mm)','rx (rad)','ry (rad)','rz (rad)'], index = self.__nodes_details.index.to_list())
        return Disppd
    
    def GlobalK(self):
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        return self.__K_Global

    def LocalK(self,element= None):
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        if element==None:
            return self.__K_Local
        else:
            pos = self.member_list.index(element)
            return self.__K_Local[pos, :, :]


    def beamsD(self):
        bd= self.beams_detail.copy()

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")       
        return(bd)

    def columnsD(self):
        cd= self.columns_detail.copy()

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")

        return(cd) 

    def nodesD(self):
        nd= self.nodes_detail.copy()

        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
    
        return(nd) 
    
    def Sdrift(self):
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
            
        if self.__seismic_def_status == False:
            return ("Seismic analysis has not been used")
        else:
            return (np.round(self.__ASD,5))
    
    def seismicS(self):
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
            

        if self.__seismic_def_status== False:
            return ("Seismic analysis has not been used")
        else:
            return (self.__SeismicShear)
        
    def memF(self):
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        return (self.__SF_BM)
    

    def maxmemF(self):
        return (self.__maxF_pd)

    def defLD(self):
        if self.__Analysis_performed== False:
            raise Exception ("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
            
        
        return (self.__Deflections_local)    
    
    def defGD(self):
        if self.__Analysis_performed== False:
            exit("Please perform the analyis of structure first by calling function 'RCAnalysis' ")
        
        return (self.__Deflections_G)
    
    def Mproperties(self):
        return (self.__Mproperties)

    def floorD(self):
        if self.__PreP_status == False:
            raise Exception("Perform Pre-Processing of the structure using method 'preP' to get complete floor details")

        if self.__slabload_there== False:
            return ("No Floor has been assigned")
        else:
            return (self.__slab_details)

    def seismicD(self):
        cols= ["Zone Factor (Z)", "Importance Factor (I)", "Response Reduction Factor (R)",  "Design Accelaration Coefficient (Sag) based on Soil Type", "Damping (%)", "Soil Type", "Time Period (sec)", "Seismic Acceleration (Ah)", "Seismic Weight (kN)" ]

        if self.__seismic_def_status== True:
            if (self.load_combo.iloc[0, 2:] >0).any():
                self.__seismicD.columns= cols
                return (self.__seismicD)

    def changeFL(self, floor=None,thickness= None, LL=None, FF=None, WP=None, delf=None ):

        if self.__PreP_status == False:
            raise Exception("Perform Pre-Processing of the structure using method 'preP' before changing floor details")

        if floor==None:
            if self.__slabload_there== True:
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
            if self.__slabload_there== True:
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
            if self.__slabload_there== True:
                self.__slab_details.drop(delf, inplace = True)

        self.slab_pd= None
        self.__slab_beam_load_transfer()

        if self.__Analysis_performed== True:
            self.__resetRC()
        
        self.__Analysis_performed= False
        

    def modelND(self):
        return (self.__ndd)
    
    def modelMD(self):
        return (self.__mdd)
    
    def modelBCD(self):
        return (self._bcd)


    def getGlobalVariables(self):
        memD= self.__member_details

        S_pd= self.slab_pd
        BN= self.baseN
        S_def= self.__seismic_def
        M_prop= self.__Mproperties
        return memD, S_pd, BN, S_def, M_prop
