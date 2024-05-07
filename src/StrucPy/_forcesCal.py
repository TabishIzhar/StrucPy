import ray
import pandas as pd
import numpy as np
import itertools as itr

@ray.remote
def _cal_internalF(mdd, ndd, nodes, dis, trans_mat, local_stiffness, lnf , q , mem_names, i, point_L ,point_loads, slabload_there, sb, load_combo, concrete_densitySlab):

    en1 =   mdd
    en2 =   mdd

    pos1 = ndd.index.get_loc(en1)
    pos2 = ndd.index.get_loc(en2)
                            
    n1= nodes[pos1,:]
    n2= nodes[pos2,:]        
    len_beam= np.sqrt(np.sum((n2-n1)**2))     # Length in mm
    L = len_beam

    d1= dis[pos1*6: (pos1*6)+6]
    d2= dis[pos2*6: (pos2*6)+6]
    ds = np.vstack((d1,d2))

    #*** FOR INTERNAL FORCES in GLOBAL CORDINATE
    T_mat= trans_mat
    kl_new= local_stiffness
    lf_new= lnf
    


    nodalForces= (kl_new@(T_mat@ds)) + lf_new

    nodalForces= nodalForces/1000        #converting into kN
    #------------------------------------------------------------------#

    member_nodes= np.zeros([1,2])


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
    

    if point_L==True: 
        
        if ((point_loads.index == mem_names[i]).any()):
            pl= (point_loads [point_loads.index == mem_names[i]])
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


    if slabload_there==1:
        if (sb.index == mem_names[i]).any():
            beam_no= (sb [sb.index == mem_names[i]])
            beam_name= beam_no.index.values[0]
            

            for k in range (len(beam_no)):
                H= beam_no.Height_l.values[k]
                Lo= beam_no.L.values[k]

                if len(beam_no)> 1:
                    w= (load_combo.iat[0,0]*((-concrete_densitySlab*(sb.slab_t[beam_name].to_list()[k])/1000) + (sb.FF[beam_name].to_list()[k])+ (sb.WP[beam_name].to_list()[k])))      #kN/m2
                    wl= (load_combo.iat[0,1]*sb.LL[beam_name].to_list()[k]) #kN/m2
                else:
                    w= (load_combo.iat[0,0]*((-concrete_densitySlab*(sb.slab_t[beam_name])/1000) + (sb.FF[beam_name])+ (sb.WP[beam_name])))      #kN/m2
                    wl= (load_combo.iat[0,1]*sb.LL[beam_name]) #kN/m2                            

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

    xx=xx.reshape((-1,1))
    mem_BMSF_val= np.around(np.hstack((xx,Vx,Vy,Vz,Mx,My,Mz)).astype(np.double),3)      
    member_nodes[0,0]= en1
    member_nodes[0,1]= en2
    
    return (mem_BMSF_val, member_nodes, ds)