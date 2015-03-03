/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSceneData.h>


hknpPhysicsSceneData::hknpPhysicsSceneData()
:	m_worldCinfo(HK_NULL)
{

}

hknpPhysicsSceneData::hknpPhysicsSceneData( hkFinishLoadedObjectFlag f )
:	hkReferencedObject(f),
	m_systemDatas(f)
{

}

hknpPhysicsSceneData::~hknpPhysicsSceneData()
{
	if (m_worldCinfo)
	{
		m_worldCinfo->removeReference();
		m_worldCinfo = HK_NULL;
	}
}

void hknpPhysicsSceneData::addWorld( const hknpWorld* world )
{
	// Add the world construction info
	if( !m_worldCinfo )
	{
		m_worldCinfo = new hknpRefWorldCinfo;
	}
	world->getCinfo( m_worldCinfo->m_info );

	// Add the world contents as one system
	hknpPhysicsSystemData* sysData = new hknpPhysicsSystemData();
	sysData->addWorld( world );
	m_systemDatas.expandOne().setAndDontIncrementRefCount( sysData );
}

hknpPhysicsSystemData* hknpPhysicsSceneData::getSystemByName( const char* name )
{
	for (int i = 0 ; i < m_systemDatas.getSize(); ++i)
	{
		if (m_systemDatas[i]->m_name == name)
		{
			return m_systemDatas[i];
		}
	}
	return HK_NULL;
}

hknpBodyId hknpPhysicsSceneData::getBodyByName(hkStringPtr name) const
{
	const int numSystems = m_systemDatas.getSize();

	int bodyIdBase = 0;
	for (int s = 0; s < numSystems; s++)
	{
		const hknpPhysicsSystemData* systemData	= m_systemDatas[s];
		const hknpBodyId bodyId = systemData->findBodyByName(name);

		if ( bodyId.isValid() )
		{
			return hknpBodyId(bodyIdBase + bodyId.value());
		}

		bodyIdBase += systemData->m_bodyCinfos.getSize();
	}

	return hknpBodyId::invalid();
}

const hknpBodyCinfo* hknpPhysicsSceneData::getBodyCinfo(hknpBodyId bodyId) const
{
	const hknpPhysicsSystemData* systemData = getBodySystem(bodyId);
	if ( systemData )
	{
		return &systemData->m_bodyCinfos[bodyId.value()];
	}

	return HK_NULL;
}

hknpBodyCinfo* hknpPhysicsSceneData::accessBodyCinfo(hknpBodyId bodyId)
{
	return const_cast<hknpBodyCinfo*>(getBodyCinfo(bodyId));
}

const hknpPhysicsSystemData* hknpPhysicsSceneData::getBodySystem(hknpBodyId& bodyId) const
{
	const int bodyIdx		= bodyId.value();
	const int numSystems	= m_systemDatas.getSize();
	for (int s = 0, bodyIdxBase = 0; s < numSystems; s++)
	{
		const hknpPhysicsSystemData* systemData	= m_systemDatas[s];

		const int numBodies	= systemData->m_bodyCinfos.getSize();
		const int idx		= bodyIdx - bodyIdxBase;

		if ( (idx >= 0) && (idx < numBodies) )
		{
			bodyId = hknpBodyId(idx);
			return systemData;
		}

		bodyIdxBase += numBodies;
	}

	return HK_NULL;
}

hknpPhysicsSystemData* hknpPhysicsSceneData::accessBodySystem(hknpBodyId& bodyId)
{
	return const_cast<hknpPhysicsSystemData*>(getBodySystem(bodyId));
}

void hknpPhysicsSceneData::tryRemoveShape(const hknpShape* shape)
{
	for (int s = m_systemDatas.getSize() - 1; s >= 0; s--)
	{
		const hknpPhysicsSystemData* systemData	= m_systemDatas[s];
		const hkArray<hknpBodyCinfo>& bodyInfos	= systemData->m_bodyCinfos;

		for (int k = bodyInfos.getSize() - 1; k >= 0; k--)
		{
			if ( bodyInfos[k].m_shape == shape )
			{
				return;	// Shape still in use, can't remove!
			}
		}
	}

	// Not used, remove!
	for (int s = m_systemDatas.getSize() - 1; s >= 0; s--)
	{
		hknpPhysicsSystemData* systemData = m_systemDatas[s];
		const int idx = systemData->m_referencedObjects.indexOf(shape);

		if ( idx >= 0 )
		{
			systemData->m_referencedObjects.removeAt(idx);
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
