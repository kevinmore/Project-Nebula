/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


//
// hkpMoppIntAabbVirtualMachine Constructor
//
hkpMoppIntAabbVirtualMachine::hkpMoppIntAabbVirtualMachine()
{
}

//
// hkpMoppIntAabbVirtualMachine Destructor
//
hkpMoppIntAabbVirtualMachine::~hkpMoppIntAabbVirtualMachine()
{
	// nothing to do
}


void hkpMoppIntAabbVirtualMachine::generateQueryFromAabb(const hkVector4& aabbMin, const hkVector4& aabbMax, hkpMoppIntAabbVirtualMachineQuery& query)
{
	const hkVector4& maxV = aabbMax;
	const hkVector4& minV = aabbMin;


	//Scales the query into 16.16 fixed precision integer format
	m_xLo = toIntMin((minV(0) - m_code->m_info.m_offset(0)) * m_code->m_info.getScale());
	m_xHi = toIntMax((maxV(0) - m_code->m_info.m_offset(0)) * m_code->m_info.getScale());

	m_yLo = toIntMin((minV(1) - m_code->m_info.m_offset(1)) * m_code->m_info.getScale());
	m_yHi = toIntMax((maxV(1) - m_code->m_info.m_offset(1)) * m_code->m_info.getScale());

	m_zLo = toIntMin((minV(2) - m_code->m_info.m_offset(2)) * m_code->m_info.getScale());
	m_zHi = toIntMax((maxV(2) - m_code->m_info.m_offset(2)) * m_code->m_info.getScale());

	query.m_xLo = (m_xLo >> 16);
	query.m_xHi = (m_xHi >> 16) + 1;

	query.m_yLo = (m_yLo >> 16);
	query.m_yHi = (m_yHi >> 16) + 1;

	query.m_zLo = (m_zLo >> 16);
	query.m_zHi = (m_zHi >> 16) + 1;


	query.m_offset_x = 0;
	query.m_offset_y = 0;
	query.m_offset_z = 0;

	//any re-offsetting will occur in the tree
	query.m_primitiveOffset = 0;							
	query.m_shift = 0;

	query.m_properties[0] = 0;

	//hkprintf("Query %x,%x : %x,%x  %x,%x\n", query.m_xLo, query.m_xHi, query.m_yLo, query.m_yHi, query.m_zLo, query.m_zHi );
	/*
	for(int p = 0; p < hkpMoppCode::MAX_PRIMITIVE_PROPERTIES; p++)
	{
		query.m_properties[p] = 0;
	}
	*/

	//now that the tempState is the currentState, we can override the currentState
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
