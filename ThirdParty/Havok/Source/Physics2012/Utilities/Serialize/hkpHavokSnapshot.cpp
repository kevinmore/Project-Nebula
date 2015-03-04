/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Serialize/hkpHavokSnapshot.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/OStream/hkOStream.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/hkBase.h>
#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Utilities/Serialize/hkpPhysicsData.h>

#include <Common/Serialize/Util/hkRootLevelContainer.h>
#include <Physics2012/Collide/Shape/Deprecated/StorageMesh/hkpStorageMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/StorageExtendedMesh/hkpStorageExtendedMeshShape.h>
#include <Physics2012/Collide/Shape/HeightField/StorageSampledHeightField/hkpStorageSampledHeightFieldShape.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>

hkpPhysicsData* HK_CALL hkpHavokSnapshot::load(hkStreamReader* reader, hkResource** allocatedData)
{
	HK_ASSERT2(0x74de3808, reader, "Null hkStreamReader pointer was passed to hkpHavokSnapshot::load");
	HK_ASSERT2(0x54c68870, allocatedData, "Null hkResource pointer was passed to hkpHavokSnapshot::load");

	// Fail quickly if null pointers were given
	if ( (!reader) || (!allocatedData) )
	{
		return HK_NULL;
	}

	if( hkResource* resource = hkSerializeUtil::load(reader) )
	{
		if( hkRootLevelContainer* container = resource->getContents<hkRootLevelContainer>() )
		{
			// first search by type
			hkpPhysicsData* data = static_cast<hkpPhysicsData*>( container->findObjectByType( hkpPhysicsDataClass.getName()) );
			if(data == HK_NULL)
			{
				// failing that, by variant name.
				const char* byName[] = { "SnapshotSave", "hkpPhysicsData", HK_NULL };
				for( int i = 0; byName[i] != HK_NULL; ++i )
				{
					data = static_cast<hkpPhysicsData*>( container->findObjectByName( byName[i] ) );
					if( data )
					{
						break;
					}
				}
			}
			if(data != HK_NULL)
			{
				*allocatedData = resource;
				return data;
			}
		}
	}

	HK_WARN(0x764219fe, "Could not load a hkRootLevelContainer from given stream.");
	return HK_NULL;
}
hkpHavokSnapshot::ConvertListener::~ConvertListener()
{
	for( int i = 0; i < m_objects.getSize(); ++i )
	{
		delete m_objects[i];
	}
}

void hkpHavokSnapshot::ConvertListener::addObjectCallback( ObjectPointer& p, ClassPointer& k )
{
	if( hkpMeshShapeClass.isSuperClass(*k) && k != &hkpStorageMeshShapeClass )
	{
		const hkpMeshShape* mesh = static_cast<const hkpMeshShape*>(p);
		hkpStorageMeshShape* storage = new hkpStorageMeshShape(mesh);
		m_objects.pushBack(storage);

		p = storage;
		k = &hkpStorageMeshShapeClass;
	}
	if( hkpExtendedMeshShapeClass.isSuperClass(*k) && k != &hkpStorageExtendedMeshShapeClass )
	{
		const hkpExtendedMeshShape* mesh = static_cast<const hkpExtendedMeshShape*>(p);
		hkpStorageExtendedMeshShape* storage = new hkpStorageExtendedMeshShape(mesh);
		m_objects.pushBack(storage);

		p = storage;
		k = &hkpStorageExtendedMeshShapeClass;
	}
	else if( hkpSampledHeightFieldShapeClass.isSuperClass(*k) )
	{
		const hkpSampledHeightFieldShape* sampled = static_cast<const hkpSampledHeightFieldShape*>(p);
		if (sampled->m_heightfieldType == hkpSampledHeightFieldShape::HEIGHTFIELD_USER)
		{
			hkpShape* storage = new hkpStorageSampledHeightFieldShape(sampled);
			m_objects.pushBack(storage);

			p = storage;
			k = &hkpStorageSampledHeightFieldShapeClass;
		}
	}
	else if( hkpRigidBodyClass.isSuperClass(*k) )
	{
		const hkpRigidBody* body = static_cast<const hkpRigidBody*>(p);
		if( hkpWorld* world = body->getWorld() )
		{
			if( world->getFixedRigidBody() == body )
			{
				p = HK_NULL;
				k = HK_NULL;
			}
		}
	}
}

hkBool HK_CALL hkpHavokSnapshot::save(const hkpWorld* world, hkStreamWriter* writer, Options outputOptions, const hkStructureLayout::LayoutRules* targetLayout, bool saveContactPoints)
{
	// Note: because hkpPhysicsData adds a ref to all rbs in the world, and removes the ref
	// on destruction, we have to:
	// Mark the world for write
	// Cast away the const of the world, so we can do this
	// Scope the hkPhysics data so that it goes out of scope and removes refs while the world
	// is still marked for write

	HK_ASSERT2(0x4bb93313, world, "Null hkpWorld pointer passed to hkpHavokSnapshot::save.");
	HK_ASSERT2(0x23ec02e2, writer, "Null hkStreamWriter passed to hkpHavokSnapshot::save.");

	// Fail if any null pointers were given
	if ( (!world) || (!writer) )
	{
		return false;
	}

	hkpWorld* mutableWorld = const_cast<hkpWorld*>(world);
	hkBool ret;
	mutableWorld->markForWrite();
	{
		// Make a data struct to contain the world info.
		hkpPhysicsData data;
		data.populateFromWorld( mutableWorld, saveContactPoints );

		ret = save( &data, writer, outputOptions, targetLayout );
	}
	mutableWorld->unmarkForWrite();

	return ret;
}

hkBool HK_CALL hkpHavokSnapshot::save( const hkpPhysicsData* data, hkStreamWriter* writer, Options outputOptions, const hkStructureLayout::LayoutRules* targetLayout)
{
	// Add this to our root level container object
	return saveUnderRootLevel(data, hkpPhysicsDataClass, writer, outputOptions, targetLayout); 
}

hkBool HK_CALL hkpHavokSnapshot::saveUnderRootLevel( const void* data, const hkClass& dataClass, hkStreamWriter* writer, Options outputOptions, const hkStructureLayout::LayoutRules* targetLayout )
{
	//assume data is the raw data, so create a named variant out of it
	hkRootLevelContainer container;
	container.m_namedVariants.expandOne().set(dataClass.getName(), const_cast<void *>(data), &dataClass);

	return save(&container, hkRootLevelContainerClass, writer, outputOptions, targetLayout);
}

hkBool HK_CALL hkpHavokSnapshot::save(
	const void* data,
	const hkClass& dataClass,
	hkStreamWriter* writer,
	Options outputOptions,
	const hkStructureLayout::LayoutRules* targetLayout,
	hkPackfileWriter::AddObjectListener* userListener )
{
	ConvertListener defaultConvertListener;
	hkPackfileWriter::AddObjectListener* convertListener = userListener
		? userListener
		: static_cast<hkPackfileWriter::AddObjectListener*>(&defaultConvertListener);
	hkSerializeUtil::SaveOptions serializeUtilOptions; serializeUtilOptions.useText( outputOptions.allAreSet(SNAPSHOT_TEXT) );

	if (outputOptions.allAreSet(SNAPSHOT_PACKFILE))
	{
		hkPackfileWriter::Options options;		
		if (targetLayout)
		{			
			options.m_layout = hkStructureLayout(*targetLayout);			
		}		
		return hkSerializeUtil::savePackfile(data, dataClass, writer, options, convertListener, serializeUtilOptions) == HK_SUCCESS;
	}
	else
	{
		return hkSerializeUtil::saveTagfile(data, dataClass, writer, convertListener, serializeUtilOptions) == HK_SUCCESS;
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
