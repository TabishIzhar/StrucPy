        xx= self.__nodes_details.to_numpy()
        nodetext= self.__nodes_details.index.to_numpy()
        xxx= self.__cords_member_order.to_numpy()
        tmm=len(xxx)
        Model = go.Figure()
        
        Model.add_trace(go.Scatter3d(x=xx[:,2],y=xx[:,0],z=xx[:,1],mode='markers+text', text=nodetext,textposition="top right"))
        kk=0
        mem_index= self.member_list
        anno=[]
        for i in range(0,tmm,2):
            
            Model.add_trace(go.Scatter3d(x=xxx[i:i+2,2],y=xxx[i:i+2,0],z=xxx[i:i+2,1], mode='lines',      
                line=dict(
                        color="black",                # set color to an array/list of desired values
                        width=10)))

            ax= xxx[i,2].item() 
            bx= xxx[i+1,2].item() 
            ay= xxx[i,0].item() 
            by= xxx[i+1,0].item() 
            az= xxx[i,1].item() 
            bz= xxx[i+1,1].item() 


            x_annotate=((ax+bx)/2)+0.1
            y_annotate=((ay+by)/2)+0.1
            z_annotate=((az+bz)/2)+0.1
            
            #(x_annotate, y_annotate, z_annotate)
            a1= dict(
            showarrow=False,
            x=x_annotate,
            y=y_annotate,
            z=z_annotate,
            text=f"Member {mem_index[kk]}",
            font=dict(
                color="black",
                size=8
            ),)

            anno.append(a1)
            kk= kk+1
        
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
                Model.add_trace(go.Surface(x=node['z'], y=node['x'], z=z,opacity=0.2,showscale=False))        

                x_an=node["z"].mean()
                y_an=node["x"].mean()
                z_an=node["y"].mean() +0.5
            
                a2= dict(
                    showarrow=False,
                    x=x_an,
                    y=y_an,
                    z=z_an,
                    text=f"Floor {i}",
                    font=dict(
                    color="black",
                    size=8
                ),)
                
                anno.append(a2)
                #floor_anno.append(a2)


        Model.update_layout(
            scene=dict(
                xaxis=dict(type="-"),
                yaxis=dict(type="-"),
                zaxis=dict(type="-"),
                annotations=anno),)

        Model.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   yaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
                                   zaxis = dict(showgrid = False,showticklabels = False,showbackground= False),
             ))

        Model.update_layout(height=800, width=1600)

        return (Model)