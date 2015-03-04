/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>
#include <Common/Base/Container/StringMap/hkStorageStringMap.h>

hkRelocationInfo::~hkRelocationInfo()
{
	delete m_pool;
}

void hkRelocationInfo::applyLocalAndGlobal( void* buffer )
{
	char* ret = static_cast<char*>(buffer);

	// apply all fixups
	{
		for( int i = 0; i < m_local.getSize(); ++i )
		{
			*(void**)(ret + m_local[i].m_fromOffset) = ret + m_local[i].m_toOffset;
		}
	}
	{
		for( int i = 0; i < m_global.getSize(); ++i )
		{
			*(void**)(ret + m_global[i].m_fromOffset) = m_global[i].m_toAddress;
		}
	}
}

void hkRelocationInfo::addImport(int off, const char* name)
{
	if( m_pool == HK_NULL )
	{
		m_pool = new hkStorageStringMap<int>();
	}
	m_imports.pushBack( Import(off, m_pool->insert(name,0) ) );
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
