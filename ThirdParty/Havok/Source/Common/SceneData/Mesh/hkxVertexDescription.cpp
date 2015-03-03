/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Mesh/hkxVertexDescription.h>

hkxVertexDescription::hkxVertexDescription(hkFinishLoadedObjectFlag f) : m_decls(f)
{

}

bool hkxVertexDescription::operator== (const hkxVertexDescription& other ) const
{
	if (m_decls.getSize() == other.m_decls.getSize())
	{
		for (int i=0; i < m_decls.getSize(); ++i)
		{
			if ( (m_decls[i].m_type != other.m_decls[i].m_type) ||
				 (m_decls[i].m_usage != other.m_decls[i].m_usage) ||
				 (m_decls[i].m_byteOffset != other.m_decls[i].m_byteOffset) ||
				 (m_decls[i].m_byteStride != other.m_decls[i].m_byteStride) ||
				 (m_decls[i].m_numElements != other.m_decls[i].m_numElements) )
			{
				return false;
			}
		}

		return true;
	}
	return false;
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
