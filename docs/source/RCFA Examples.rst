RCFA Examples
=============

* Need to install **openpyxl** to read excel file.
* Input files are available for every example as download.
* Path for Input Files must be correct, and must be adjusted as per user location for every example.
* Location for the input files are directly in the directory of test script.

Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.
---------------------------------------------------------------------------------------------
See :download:`Input Example1 <./Examples/Example1.xlsx>`.

.. code-block:: python

    from StrucPy.RCFA import RCF
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    

    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('./Example1.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('./Example1.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('./Example1.xlsx', 'boundary', header = 0, index_col=0)
    point_loads= pd.read_excel('./Example1.xlsx', 'point_loads', header = 0, index_col=0)

    # Self weight is being ignored
    r1= RCF(nodes_details,member_details,boundcond, point_loads= point_loads, self_weight=False)

    #Pre processing the model
    r1.preP()

    # Performing Analysis
    r1.RCanalysis()

    # Getting Reactions
    reactions= r1.reactions()

    # Getting Shear Force and Bending Moment Diagram of Member with ID- 1
    sfbmd= r1.sfbmd(1)

    # Getting Deflected Shape in Local Coordinate System of Member with ID- 1
    defL= r1.defL(1)

    # Getting Deflected Shape in Global Coordinate System of Member with ID- 1
    defG= r1.defG(1)

    # Getting Material Properties
    material_properties= r1.Mproperties()



Example 2: Analysis of simple supported beam with uniformly distributed loads (UDL) using :class:`StrucPy.RCFA.RCF`.
--------------------------------------------------------------------------------------------------------------------
See :download:`Input Example2 <./Examples/Example2.xlsx>`.

.. code-block:: python

    from StrucPy.RCFA import RCF
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go


    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('./Example2.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('./Example2.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('./Example2.xlsx', 'boundary', header = 0, index_col=0)
    point_loads= pd.read_excel('./Example2.xlsx', 'point_loads', header = 0, index_col=0)

    # Check "boundcond" for defining different boundary condition

    # Self weight is being ignored
    r1= RCF(nodes_details,member_details,boundcond, self_weight=False)

    #Pre processing the model
    r1.preP()

    # Performing Analysis
    r1.RCanalysis()

    # Getting Reactions
    reactions= r1.reactions()

    # Getting Shear Force and Bending Moment Diagram of Member with ID- 1
    sfbmd= r1.sfbmd(1)

    # Getting Deflected Shape in Local Coordinate System of Member with ID- 1
    defL= r1.defL(1)

    # Getting Deflected Shape in Global Coordinate System of Member with ID- 1
    defG= r1.defG(1)

    # Getting Material Properties
    material_properties= r1.Mproperties()



Example 3: Analysis of 8-story regular building with UDL of -50kN/m on all beams using :class:`StrucPy.RCFA.RCF`. Self-weight not considered.
---------------------------------------------------------------------------------------------------------------------------------------------
See :download:`Input Example3 <./Examples/Example3.xlsx>`.

.. code-block:: python

    from StrucPy.RCFA import RCF
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go


    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('./Example3.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('./Example3.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('./Example3.xlsx', 'boundary', header = 0, index_col=0)

    # Check "boundcond" for defining different boundary condition

    # Self weight is being ignored
    r1= RCF(nodes_details,member_details,boundcond, self_weight=False)

    #Pre processing the model
    r1.preP()

    # Performing Analysis
    r1.RCanalysis()


    # View 3D Model of Structure
    view_3Dmodel= r1.model3D()

    view_3Dmodel.show()                                     # If using VS Code or JupiterNotebook
    view_3Dmodel.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Reactions
    base_reactions= r1.reactions()


    # Getting Nodal Displacements
    nodal_displacements= r1.Gdisp()


    # View 3D Deflected Shape of Structure
    view_structure_deflected_shape= r1.def3D()

    view_structure_deflected_shape.show()                              # If using VS Code or JupiterNotebook
    view_structure_deflected_shape.write_html("./model3D.html")       # To save model as html in C drive


    # To generate 3D Deflection Animation of Structure
    view_deflection_animation= r1.aniDef()

    view_deflection_animation.show()                                     # If using VS Code or JupiterNotebook
    view_deflection_animation.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Shear Force and Bending Moment Diagram of Member with ID- 756
    sfbmd= r1.sfbmd(756)

    sfbmd.show()                                     # If using VS Code or JupiterNotebook
    sfbmd.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Deflected Shape in Local Coordinate System of Member with ID- 756
    defL= r1.defL(756)

    defL.show()                                     # If using VS Code or JupiterNotebook
    defL.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Deflected Shape in Global Coordinate System of Member with ID- 756
    defG= r1.defG(756)

    defG.show()                                     # If using VS Code or JupiterNotebook
    defG.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Material Properties
    material_properties= r1.Mproperties()

    # Getting Details of All Beam Members
    beams_details= r1.beamsD()

    # Getting Details of All Column Members
    columns_details= r1.columnsD()

    # Getting Details of All Nodes (Joints)
    nodes_deatils= r1.nodesD()

    # Getting Data Details of All Member Forces in Every Member
    member_forces_SF_BM_in_all_direction= r1.memF()

    # Getting Data Details of Maximum Forces in Every Member
    max_member_forces_in_all_direction= r1.MaxmemF()

    # Getting Data Details of Deflection of Every Member in Local Coordinate System
    deflection_local_coordinate_data= r1.defLD()

    # Getting Data Details of Deflection of Every Member in Global Coordinate System
    deflection_global_coordinate_data= r1.defGD()

    # Getting Global Stiffness Matrix of Structure in Global Coordinate System
    get_Global_stiffness_of_structure= r1.GlobalK()    


Example 4: Analysis of 5-story irregular building with floor loads and self weight using :class:`StrucPy.RCFA.RCF`. Floor Loads generated using :class:`StrucPy.RCFA.RCF.autoflooring`, and changes are made using inbuilt method :class:`StrucPy.RCFA.RCF.changeFL`. Load Combination (1.5 DL + 1.5 LL) is used.
--------------------------------------------------------------------------------------------------------------------------------------------------------
See :download:`Input Example4 <./Examples/Example4.xlsx>`.

.. code-block:: python

    from StrucPy.RCFA import RCF
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go


    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('./Example4.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('./Example4.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('./Example4.xlsx', 'boundary', header = 0, index_col=0)
    load_combo= pd.read_excel('./Example4.xlsx', 'load_combinations', header = 0, index_col=0)

    # Check "boundcond" for defining different boundary condition

    #Creating RCF Object for structure
    r1= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True)

    #Pre processing the model
    r1.preP()

    # View Floor Details (includes floor thickness, Floor Finish Loads, Live Loads and Water proofing Loads in kN/m2)
    floor_load_details= r1.floorD()


    # Changing all floor loads with  Live Load as 50kN/m2 and Floor thickness of 1000mm
    r1.changeFL(thickness= 1000, LL=-50, FF=0)

    # Performing Analysis
    r1.RCanalysis()

    # View 3D Model of Structure
    view_3Dmodel= r1.model3D()

    view_3Dmodel.show()                                     # If using VS Code or JupiterNotebook
    view_3Dmodel.write_html("./model3D.html")              # To save model as html in C drive

    # Getting Reactions
    base_reactions= r1.reactions()


    # Getting Nodal Displacements
    nodal_displacements= r1.Gdisp()


    # View 3D Deflected Shape of Structure
    view_structure_deflected_shape= r1.def3D()

    view_structure_deflected_shape.show()                              # If using VS Code or JupiterNotebook
    view_structure_deflected_shape.write_html("./model3D.html")       # To save model as html in C drive


    # To generate 3D Deflection Animation of Structure
    view_deflection_animation= r1.aniDef()

    view_deflection_animation.show()                                     # If using VS Code or JupiterNotebook
    view_deflection_animation.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Shear Force and Bending Moment Diagram of Member with ID- 756
    sfbmd= r1.sfbmd(12)

    sfbmd.show()                                     # If using VS Code or JupiterNotebook
    sfbmd.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Deflected Shape in Local Coordinate System of Member with ID- 756
    defL= r1.defL(12)

    defL.show()                                     # If using VS Code or JupiterNotebook
    defL.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Deflected Shape in Global Coordinate System of Member with ID- 756
    defG= r1.defG(12)

    defG.show()                                     # If using VS Code or JupiterNotebook
    defG.write_html("./model3D.html")              # To save model as html in C drive


    # Getting Material Properties
    material_properties= r1.Mproperties()

    # Getting Details of All Beam Members
    beams_details= r1.beamsD()

    # Getting Details of All Column Members
    columns_details= r1.columnsD()

    # Getting Details of All Nodes (Joints)
    nodes_deatils= r1.nodesD()

    # Getting Data Details of All Member Forces in Every Member
    member_forces_SF_BM_in_all_direction= r1.memF()

    # Getting Data Details of Maximum Forces in Every Member
    max_member_forces_in_all_direction= r1.MaxmemF()

    # Getting Data Details of Deflection of Every Member in Local Coordinate System
    deflection_local_coordinate_data= r1.defLD()

    # Getting Data Details of Deflection of Every Member in Global Coordinate System
    deflection_global_coordinate_data= r1.defGD()

    # Getting Global Stiffness Matrix of Structure in Global Coordinate System
    get_Global_stiffness_of_structure= r1.GlobalK()    


Example 5: Static seismic analysis of a 5-story irregular building with floor loads and self weight using :class:`StrucPy.RCFA.RCF`. Floor Loads generated using method autoflooring, and changes are made using inbuilt method. Seismic force is applied in x-direction. Load Combination is used: 1.5 DL + 1.2 EQx.
------------------------------------------------------------------------------------------------------------------------------------------------
See :download:`Input Example5 <./Examples/Example5.xlsx>`.

.. code-block:: python

    from StrucPy.RCFA import RCF
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go


    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('./Example5.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('./Example5.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('./Example5.xlsx', 'boundary', header = 0, index_col=0)
    load_combo= pd.read_excel('./Example5.xlsx', 'load_combinations', header = 0, index_col=0)
    seismic_defination= pd.read_excel('./Example5.xlsx', 'Seismic_Defination', header = 0, index_col=0)

    # Check "seismic_defination" for defining seismic defination

    # Creating RCF object for structure analysis
    r1= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination)

    #Pre processing the model
    r1.preP()

    # View Floor Details (includes floor thickness, Floor Finish Loads, Live Loads and Water proofing Loads in kN/m2)
    floor_load_details= r1.floorD()


    # Changing all floor loads with Floor Finish load as 50kN/m2
    r1.changeFL(thickness= 0, LL=0, FF=-50)

    # Performing Analysis
    r1.RCanalysis()

    # View 3D Model of Structure
    view_3Dmodel= r1.model3D()

    # Getting Reactions
    base_reactions= r1.reactions()

    # Getting Nodal Displacements
    nodal_displacements= r1.Gdisp()

    #Getting details of seismic calculation.
    seismicD= r1.seismicD()

    #Getting details of seismic shear.
    seismic_shear= r1.seismicS()

    #Getting details of stroy drift.
    drift= r1.Sdrift()    


Example 6: Changing the material properties of members of "RCFA Examples:Example 5".
-----------------------------------------------------------------------------------------
See :download:`Input Example6 <./Examples/Example6.xlsx>`.

.. code-block:: python

    from StrucPy.RCFA import RCF
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go


    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('../Examples/InputFiles/Example6.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('../Examples/InputFiles/Example6.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('../Examples/InputFiles/Example6.xlsx', 'boundary', header = 0, index_col=0)
    load_combo= pd.read_excel('../Examples/InputFiles/Example6.xlsx', 'load_combinations', header = 0, index_col=0)
    seismic_defination= pd.read_excel('../Examples/InputFiles/Example6.xlsx', 'Seismic_Defination', header = 0, index_col=0)

    grade_concrete= 30              # If want same calculation but just different concrete grade, grade_concrete can be used.

    material_properties= pd.read_excel('../Examples/InputFiles/Example6.xlsx', 'Mproperties', header = 0, index_col=0)


    # Check "seismic_defination" for defining seismic defination

    # Creating RCF object "r1" for structure analysis just changing the grade of concrete, entire calculation for estimation of young modulus, modulus of rigidity etc. remains same 
    r1= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination, grade_conc= grade_concrete)


    # Creating RCF object "r2" for structure analysis, completely changing the material properties
    r2= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination, properties= material_properties )


    #Pre processing the model
    r1.preP()

    r2.preP()

    # View Floor Details (includes floor thickness, Floor Finish Loads, Live Loads and Water proofing Loads in kN/m2)
    floor_load_details= r1.floorD()


    # Changing all floor loads with Floor Finish load as 50kN/m2
    r1.changeFL(thickness= 0, LL=0, FF=-50)

    r2.changeFL(thickness= 0, LL=0, FF=-50)


    # Performing Analysis
    r1.RCanalysis()
    r2.RCanalysis()



    # Getting Reactions
    base_reactions_r1= r1.reactions()
    base_reactions_r2= r2.reactions()

    #Getting Material Properties
    Mproperties_r1= r1.Mproperties()

    Mproperties_r2= r2.Mproperties()

Example 7: Analysis of 5-story irregular building with floor loads and self weight for multiple load combinations using :class:`StrucPy.RCFA.RCFenv`. Floor Loads generated using method autoflooring, and changes are made using inbuilt method.
---------------------------------------------------------------------------------------------------------------------------------------------
See :download:`Input Example7 <./Examples/Example7.xlsx>`.

.. code-block:: python

    from StrucPy.RCFA import RCFenv
    import pandas as pd
    import plotly
    import ray


    # Importing Input Data from Excel File (Note: Change the Path as per the location of File)
    member_details= pd.read_excel('../Examples/InputFiles/Example7.xlsx', 'members', header = 0, index_col=0)
    nodes_details= pd.read_excel('../Examples/InputFiles/Example7.xlsx', 'nodes', header = 0, index_col=0)
    boundcond = pd.read_excel('../Examples/InputFiles/Example7.xlsx', 'boundary', header = 0, index_col=0)
    load_combo= pd.read_excel('../Examples/InputFiles/Example7.xlsx', 'load_combinations', header = 0, index_col=0)
    seismic_defination= pd.read_excel('../Examples/InputFiles/Example7.xlsx', 'Seismic_Defination', header = 0, index_col=0)


    # Creating RCFenv object "r1" for structure analysis
    r1= RCFenv(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination)

    #Pre processing the model
    r1.preP()

    # Changing all floor loads with Floor Finish load as 50kN/m2
    r1.changeFL(thickness= 0, LL=-25, FF=-50)

    # Performing Analysis
    r1.RCanalysis()

    # Getting Reactions for every load combinations at base nodes 
    base_reactions= r1.getReact()

    # Getting Maximum Reactions at base nodes
    base_max_reactions= r1.getReactmax()

    # Getting nodal displacement for every load combinations at all nodes
    nodal_displacements= r1.getNdis()

    # Getting max nodal displacement at all nodes
    nodal_max_displacements= r1.getNdismax()

    # Getting member forces at extreme end of all members for every load combinations
    end_member_forces= r1.getEndMF()

    # Getting maximum positive and negative forces in all members for every load combinations
    end_member_forces_max= r1.getMFmax()

    # Getting maximum (design) forces in all members 
    design_member_forces= r1.getMFdsg()

    # Getting members displacement data for every load combinations in local coordinate system.
    member_displacements_local= r1.getLDef()

    # Getting members displacement data for every load combinations in global coordinate system.
    member_displacements_global= r1.getGDef()
    
    
    # Getting maximum displacement data for members in local coordinate system.
    member_displacements_local= r1.getLDefmax()

    # Getting maximum displacement data for members in global coordinate system.
    member_displacements_global= r1.getGDefmax()

    # Retrieving all the objects corresponding to different load combinations
    LCobj= r1.getTLC()


    #All the method of class RCF can be accessed for all the objects corresponding to different load combinations

    LC1= LCobj[0]
    LC2= LCobj[1]
    LC3= LCobj[2]
    LC4= LCobj[3]    #..so no upto LC11= LCobj[10]

    #Accessing 3D model, 3D model will be same for every objects
    model_3D = ray.get(LC1.model3D.remote())
    model_3D.show()

    #Accessing deflected 3D shape of model for LC1 and LC4
    def_model_3D_LC1 = ray.get(LC1.def3D.remote())
    def_model_3D_LC4 = ray.get(LC4.def3D.remote())

    def_model_3D_LC1.show()
    def_model_3D_LC4.show()


    