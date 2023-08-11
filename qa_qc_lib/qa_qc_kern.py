import numpy as np

class QA_QC_kern():
    def __init__(self, file_path:str) -> None:
        """_summary_

        Args:
            data (str): _description_
        """ 
        pass


    def first_test(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        return True


    def second_test(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        return True


    def get_list_of_tests(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """        
        return ['first_test', 'second_test']


    def start_tests(self, list_of_tests:list) -> None:
        """_summary_

        Args:
            list_of_tests (list): _description_
        """        
        pass


    def generate_test_report(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """        
        return 'test results'
    
        def test_saturation(self, Archi_model= None, J_model = None, ofp_model = None, gis_type: str = 'ild', poro_model = None, poroeff_model = None, perm_model = None, gis_type1: str= 'rhob', gis_type2: str=None):

        rigis = self.properties(rigis = self.properties(poro_model = poro_model, poroeff_model = poroeff_model, perm_model = perm_model, gis_type1 = gis_type1, gis_type2=gis_type2))
        
        if not poro_model or not J_model or not Archi_model or not ofp_model:
            print('Введите все запрашиваемые модели свойств')
            pass 
        

        saturation = pd.DataFrame(columns=['poro', 'Archi', 'J', 'ofp', 'depth'])
        depth = self.gis['depth']
        gis_for_properties = self.gis[gis_type]
        kern_poro = []
        rigis_poro = []
            
        for i in range(1, len(self.depth)):
            if self.depth[i] > 0:
                    depthindex = np.where(depth == round(self.depth[i], 2))[0][0]
                    if gis_for_properties[depthindex] > 0 and self.poro_open[i] > 0:
                        o = len(saturation['poro'])
                        saturation.at[o, 'Archi'] = Archi_model(gis_for_properties[depthindex], rigis.loc[i]['poro'])
                        saturation.at[o, 'depth'] = depth[depthindex]
                        saturation.at[o, 'J'] = J_model(rigis.loc[o]['poro'])
                        saturation.at[o, 'ofp'] = ofp_model(rigis.loc[o]['poro'])
                        saturation.at[o, 'poro'] = rigis.loc[o]['poro']
        
        for i in saturation.columns()[1:4]:
            print(i)

    
        if [x for x in rigis_poro if x>1]:
            rigis_poro = [x/100 for x in rigis_poro]
        if [x for x in kern_poro if x>1]:
            kern_poro = [x/100 for x in kern_poro]

        return rigis, np.array(kern_poro), np.array(rigis_poro)