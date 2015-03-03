/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#if defined(HK_PLATFORM_SPU)
#	include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>	// for HK_MAX_NUM_HITS_PER_AABB_QUERY define
#endif
void hkpMoppVirtualMachine::addHit(unsigned int id, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES])
{
/*
	switch ( hkpMoppCode::MAX_PRIMITIVE_PROPERTIES )
	{
	//case 3: info.properties[2] = properties[2];
	//case 2:	info.properties[1] = properties[1];
	//case 1:	info.properties[0] = properties[0];
	case 0:	break;
	//default: HK_ASSERT(0x2387d53b, 0);
	}
*/
#if !defined(HK_PLATFORM_SPU)
	
	m_primitives_out->expandOne().ID = id; 
#else
	if ( m_primitives_idx < m_primitives_out_capacity )
	{
		m_primitives_out[ m_primitives_idx ].ID = id ;
		m_primitives_idx = m_primitives_idx + 1;
	}
#endif

	HK_ASSERT2(0x6dcad53c, properties[0] == 0, "This MOPP code format has been deprecated. You need to rebuild your MOPP code.");

#ifdef HK_MOPP_DEBUGGER_ENABLED
	if ( hkpMoppDebugger::getInstance().find() )	{	hkprintf("Adding correct triangle as %i %i\n", id, properties[0]);	}
#endif
}

#if !defined(HK_PLATFORM_SPU)

void hkpMoppVirtualMachine::initQuery( hkArray<hkpMoppPrimitiveInfo>* primitives_out )
{
	hkpMoppVirtualMachine::m_primitives_out = primitives_out;
}

#else

void hkpMoppVirtualMachine::initQuery( hkpMoppPrimitiveInfo* primitives_out, int primitives_out_capacity )
{
	m_primitives_out = primitives_out;
	m_primitives_idx = 0;
	m_primitives_out_capacity = primitives_out_capacity;
}

#endif

hkpMoppVirtualMachine::hkpMoppVirtualMachine()
{

}
hkpMoppVirtualMachine::~hkpMoppVirtualMachine()
{

}

int HK_CALL hkpMoppVirtualMachine::toIntMin(hkReal x)
{
	return hkMath::hkToIntFast(x)-1;
}

int HK_CALL hkpMoppVirtualMachine::toIntMax(hkReal x)
{
	return hkMath::hkToIntFast(x)+1;
}


inline int HK_CALL hkpMoppVirtualMachine::read24( const unsigned char* PC )
{
	return (PC[0]<<16) + (PC[1]<<8) + PC[2];
}

inline int hkpMoppVirtualMachine::getNumHits() const
{
#if !defined(HK_PLATFORM_SPU)
	return m_primitives_out->getSize();
#else
	return m_primitives_idx;
#endif
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
