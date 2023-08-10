import ray
import pandas as pd
import numpy as np
import itertools as itr

@ray.remote
def arrange_col_Frame(md, nd):

    mdd= md.copy()
    ndd= nd.copy()

    col_mat= mdd.loc[mdd['Type']=='Col']
    col_mat = col_mat.filter(['Node1', 'Node2'])   

    p_y= ndd.sort_values(by=['y','x', 'z'])
    yy = p_y.y.unique()
    len_yy= len(yy)

    min_xyz= p_y.y.min()
    base= p_y[p_y.y.isin([min_xyz])] 
    baseN= base
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

    L= np.sqrt(np.square(ndd.loc[col_mat["Node1"]].values- ndd.loc[col_mat["Node2"]].values).sum(axis=1)) 

    col_mat.insert(2, 'L', L)
    col_mat.insert(2, 'Story', ' ')
    for i in range(len_yy-1):
        col_mat.loc[col_mat.index.isin(column_detail[f'C{i+1}']), 'Story']= i

    columns_detail= col_mat.copy()
    #COLUMN_DETAIL CONTAINS ALL NODES AND THEIR CONTINUOUS COLUMN 
    #col_mat CONTAINS general details of each COLUMN 
    return (baseN, col_mat,columns_detail)    

@ray.remote
def arrange_beam_FrameX(b_mat, nd):

    beams_mat= b_mat.copy()
    ndd= nd.copy()

    # Beam Arrangement x-direction
    b_y= ndd.sort_values(by=['y','x', 'z'])

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
        beam_detail_x.loc[i,['L']]= np.sqrt(np.square((ndd.loc[pos1]- ndd.loc[pos2])).sum())
    
    return beam_detail_x

#------------------------------------------------------------------------------
    # Beam Arrangement in z direction  
    #                    
@ray.remote
def arrange_beam_FrameZ(b_mat, nd):

    beams_mat= b_mat.copy()
    ndd= nd.copy()

    b_y= ndd.sort_values(by=['y','x', 'z'])

    xx = b_y.x.unique()
    yyb= b_y.y.unique()
    index_beam_mat= beams_mat.index

    d3= 1
    if len(yyb)== 1:     # Deals with only single beam model
        d3= 0

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
        beam_detail_z.loc[i,['L']]= np.sqrt(np.square((ndd.loc[pos1]- ndd.loc[pos2])).sum())         
  
    return (beam_detail_z)

@ray.remote
def arrange_nodes_Frame(md, nd,b_mat, bd_x, bd_z, c_mat ):

    mdd= md.copy()
    beams_mat= b_mat.copy()
    ndd= nd.copy()
    beam_detail_x  = bd_x.copy()
    beam_detail_z= bd_z.copy()
    col_mat= c_mat.copy()

    p_y1= ndd.sort_values(by=['y','x', 'z'])

    node_details= pd.DataFrame()
    node_details[['Node']]= ndd.index.values.reshape((-1,1))
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
    
    beam_details= pd.concat([beam_detail_x, beam_detail_z])
    
    real_beam_details= beam_details.copy()

    nodesD= node_details.copy()

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
            c1_d= mdd.at[c1,'d']
            c1_b= mdd.at[c1,'b']
        
        if c2==0:
            c2_d= 10
            c2_b= 10
        else:
            c2_d= mdd.at[c2,'d']
            c2_b= mdd.at[c2,'b']

        if c3==0:
            c3_d= 10
            c3_b= 10
        else:
            c3_d= mdd.at[c3,'d']
            c3_b= mdd.at[c3,'b']

        if c4==0:
            c4_d= 10
            c4_b= 10
        else:
            c4_d= mdd.at[c4,'d']
            c4_b= mdd.at[c4,'b']

        if (beam_detail_x.index.isin([i]).any()== True ):
            beam_details.at[i,"WDedn1"]= np.where((c1_d >= c2_d), c1_d, c2_d)

            beam_details.at[i,"WDedn2"]= np.where((c3_d >= c4_d), c3_d, c4_d)

        if (beam_detail_z.index.isin([i]).any()== True ):
            beam_details.at[i,"WDedn1"]= np.where((c1_b >= c2_b), c1_b, c2_b)

            beam_details.at[i,"WDedn2"]= np.where((c3_b >= c4_b), c3_b, c4_b)

    bd_LDeduct= beam_details.filter(['L', 'WDedn1', 'WDedn2'])
    
    bd_LDeduct["L_clear"]= bd_LDeduct["L"]-((bd_LDeduct['WDedn1']/2000)+(bd_LDeduct['WDedn2']/2000))

    return (bd_LDeduct, real_beam_details, beam_details, nodesD, node_details)


#------------------------------------------------------------------------------
