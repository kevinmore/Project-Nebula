/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Dynamics/SaveContactPoints/hkpSerializedAgentNnEntry.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

//__________________________________________________________________________________________________
//
// Serialized entry
//__________________________________________________________________________________________________

hkpSerializedAgentNnEntry::~hkpSerializedAgentNnEntry()
{
	// this member needs to be as large as the largest size
	HK_COMPILE_TIME_ASSERT( sizeof(m_nnEntryData) >= HK_AGENT3_MAX_AGENT_SIZE );

	if (m_bodyA) { m_bodyA->removeReference(); }
	if (m_bodyB) { m_bodyB->removeReference(); }
}


hkpSerializedTrack1nInfo::~hkpSerializedTrack1nInfo()
{
	if( (m_sectors.getCapacityAndFlags() & hkArray<char>::DONT_DEALLOCATE_FLAG) == 0)
	{
		for (int s = 0; s < m_sectors.getSize(); s++)
		{
			delete m_sectors[s];
		}
	}
	
	if( (m_subTracks.getCapacityAndFlags() & hkArray<char>::DONT_DEALLOCATE_FLAG) == 0)
	{
		for (int t = 0; t < m_subTracks.getSize(); t++)
		{
			hkpSerializedSubTrack1nInfo* subTrack = m_subTracks[t];
			delete subTrack;
		}
	}	
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
