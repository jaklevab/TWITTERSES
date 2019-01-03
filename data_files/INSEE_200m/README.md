# Grided data at 200m resolution
Data extracted from [INSEE 2016](https://www.insee.fr/fr/statistiques/2520034):  This database consists of 18 variables regarding age structure and household features over the grided French metropolitan territory in 2010. In order to uphold current privacy laws, no statistical information of grid cells of less than 11 households is diffused. Other sensitive variables were also treated. You can find the full documentation [here](https://www.insee.fr/fr/statistiques/fichier/2520034/documentation-complete-donnees-a-200m-1.pdf) (French.)

Each folder contains a dBase (DBF) file containing the demographical information as well as a MapInfo (MIF or MID) file containing the geographical contours of each grid cell. 

. **Square grided data: 200m-carreaux-metropole**
- **Id**: Identifier of the inhabited grid cell 
- **idINSPIRE**: Identifier of the inhabited grid cell 
- **idk**: Identifier of the rectangular cell containing the grid cell 
- **ind_c**: Number of individuals living in the grid cell
- **nbcar**: Number of inhabited grid cells contained in the rectangular cell

. **Rectangular grided data: 200m-rectangle-metropole**
- **idk**: Identifier of the rectangular cell 
- **men**: Number of households within the rectangular cell
- **men_surf**: Cumulative surface of households (square meters)
- **men_occ5**: Number of households living for more than 5 years in their current home
- **men_coll**: Number of households living in a shared place
- **men_5ind**: Number of households consisting of 5 or more individuals
- **men_1ind**: Number of households consisting of a single individual
- **i_1ind**: Indicator of privacy treatment of variable **men_1ind**
- **men_prop**: Number of owner households
- **i_1ind**: Indicator of privacy treatment of variable **men_prop**
- **men_basr**: Number of households living under the line of poverty
- **i_basr**: Indicator of privacy treatment of variable **men_basr**
- **ind_r**: Number of individuals living in the rectangular cell
- **ind_age1**: Number of individuals aged from 0 to 3 years old
- **ind_age2**: Number of individuals aged from 4 to 5 years old
- **ind_age3**: Number of individuals aged from 6 to 10 years old
- **ind_age4**: Number of individuals aged from 11 to 14 years old
- **ind_age5**: Number of individuals aged from 15 to 17 years old
- **ind_age6**: Number of individuals older than 25 years old
- **ind_age7**: Number of individuals older than 65 years old
- **ind_age8**: Number of individuals older than 75 years old
- **i_age7**: Indicator of privacy treatment of variable **ind_age7**
- **i_age8**: Indicator of privacy treatment of variable **ind_age8**
- **ind_age1**: Number of individuals aged from 0 to 3 years old
- **ind_srf**: Sum of fiscal income per consumption unit winsorized of individuals
- **nbcar**: Number of inhabited grid cells contained in the rectangular cell

**NB**: All geographical information uses the  Lambert Azimutal Equal Area (EPSG 3035) projection system.

