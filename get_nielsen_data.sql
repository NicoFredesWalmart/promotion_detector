SELECT NMP.MasterPrice_Dt
	, NMP.Tienda_Number
	, NMP.Barcode
	, NMP.Precio
	, VM.Cod_Categoria
FROM PR_WMCLBI.dbo.FT_NIELSEN_MASTER_PRICE AS NMP
INNER JOIN PR_WMCLBI.dbo.DM_ITEM AS DM
ON CAST(SUBSTRING(CAST(NMP.Barcode AS varchar), 1, 12) AS bigint) = DM.Upc_Nbr
INNER JOIN PR_WMCLBI.dbo.VM_DM_ITEM AS VM
ON DM.Item_Nbr = VM.Item_Nbr
WHERE NMP.MasterPrice_Dt BETWEEN CAST(DATEADD(DAY, -14, getdate()) AS date) AND CAST(getdate() AS date)
	AND Formato NOT IN ('Lider Express', 'LIDER INTERNET', 'LIDER EXPRESS', 'LIDER', 'HIPER LIDER', 'Hiper L?der', 'EXPRESS DE LIDER', 'EKONO', 'CL_LIDER_EXPRESS', 'CL_LIDER_', 'CL_LIDER', 'CL_EKONO', 'CL_ACUENTA', 'ACUENTA', '1321:Lider ')
	AND DM.Upc_Nbr IN (
		SELECT IT.Upc_Nbr 
		FROM PR_WMCLBI.dbo.FT_SALES AS SA
		INNER JOIN PR_WMCLBI.dbo.DM_ITEM AS IT
		ON SA.Item_Nbr = IT.Item_Nbr 
		INNER JOIN PR_WMCLBI.dbo.VM_DM_ORGANIZATION AS ORG
		ON SA.Store_Nbr = ORG.Store_Nbr
		WHERE SA.POS_Unit_Qty > 0
  			AND SA.POS_Sales_Amt > 0
    		AND IT.Account_Nbr IS NULL
    		AND ORG.Cod_Formato = 815
    	)