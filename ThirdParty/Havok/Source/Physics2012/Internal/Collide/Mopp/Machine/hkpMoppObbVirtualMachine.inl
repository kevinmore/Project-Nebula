/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


//
// hkpMoppObbVirtualMachine Constructor
//
hkpMoppObbVirtualMachine::hkpMoppObbVirtualMachine()
{
}

//
// hkpMoppObbVirtualMachine Destructor
//
hkpMoppObbVirtualMachine::~hkpMoppObbVirtualMachine()
{
	// nothing to do
}


void hkpMoppObbVirtualMachine::generateQueryFromAabb(const hkVector4& aabbMin, const hkVector4& aabbMax, hkpMoppObbVirtualMachineQuery& query)
{
	const hkpMoppCode::CodeInfo& info = m_code->m_info;
#if !defined(HK_PLATFORM_SPU)
	const hkVector4& maxV = aabbMax;
	const hkVector4& minV = aabbMin;

	//Scales the query into 16.16 fixed precision integer format

	hkReal scale = info.getScale();

	hkReal offset0 = info.m_offset(0);
	m_xLo = toIntMin((minV(0) - offset0 ) * scale);
	m_xHi = toIntMax((maxV(0) - offset0 ) * scale);
	hkReal offset1 = info.m_offset(1);
	m_yLo = toIntMin((minV(1) - offset1) * scale);
	m_yHi = toIntMax((maxV(1) - offset1) * scale);

	hkReal offset2 = info.m_offset(2);
	m_zLo = toIntMin((minV(2) - offset2) * scale);
	m_zHi = toIntMax((maxV(2) - offset2) * scale);
#else

	hkVector4 offset = info.m_offset;
	hkVector4 maxV; maxV.setSub( aabbMax, offset );
	hkVector4 minV; minV.setSub( aabbMin, offset );

	//Scales the query into 16.16 fixed precision integer format
	hkSimdReal scale = hkSimdReal::fromFloat(info.getScale());
	maxV.mul( scale );
	minV.mul( scale );

	m_xLo = toIntMin(minV(0));
	m_xHi = toIntMax(maxV(0));
	m_yLo = toIntMin(minV(1));
	m_yHi = toIntMax(maxV(1));
	m_zLo = toIntMin(minV(2));
	m_zHi = toIntMax(maxV(2));
#endif

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

	/*
	for(int p = 0; p < hkpMoppCode::MAX_PRIMITIVE_PROPERTIES; p++)	{		query.m_properties[p] = 0;	}
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
