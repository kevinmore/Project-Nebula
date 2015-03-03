/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkLoader.h>

#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Serialize/Packfile/hkPackfileData.h>
#include <Common/Serialize/Util/hkRootLevelContainer.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>

hkLoader::~hkLoader()
{
	// We destruct all of the objects first in case the objects in one
	// packfile refer to those in another.
	for (int i=0; i < m_loadedData.getSize(); i++)
	{
		if (m_loadedData[i]->getReferenceCount() == 1)
		{
			m_loadedData[i]->callDestructors();
		}
	}
	for (int i=0; i < m_loadedData.getSize(); i++)
	{
		m_loadedData[i]->removeReference();
	}
	m_loadedData.setSize(0);
}

hkRootLevelContainer* hkLoader::load( const char* filename )
{
	return load( filename, hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry() );
}

hkRootLevelContainer* hkLoader::load( hkStreamReader* streamIn )
{
	return load( streamIn, hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry() ); 
}

hkRootLevelContainer* hkLoader::load( const char* filename, hkTypeInfoRegistry* finish )
{
	return static_cast<hkRootLevelContainer*>( load( filename, hkRootLevelContainerClass, finish ) );
}

hkRootLevelContainer* hkLoader::load( hkStreamReader* streamIn, hkTypeInfoRegistry* finish )
{
	return static_cast<hkRootLevelContainer*>( load( streamIn, hkRootLevelContainerClass, finish ) );
}

void* hkLoader::load( const char* filename, const hkClass& expectedTopLevelClass )
{
	return load(filename,
		expectedTopLevelClass,
		hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry() );
}

void* hkLoader::load( hkStreamReader* reader, const hkClass& expectedTopLevelClass )
{
	return load(reader,
		expectedTopLevelClass,
		hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry() );
}

void* hkLoader::load( const char* filename, const hkClass& expectedClass, hkTypeInfoRegistry* finish )
{
	hkIstream fileIn(filename);
	if (fileIn.isOk())
	{
		return load( fileIn.getStreamReader(), expectedClass, finish );
	}
	
	HK_WARN(0x5e543234, "Unable to open file " << filename);
	return HK_NULL;
}

void* hkLoader::load( hkStreamReader* streamIn, const hkClass& expectedClass, hkTypeInfoRegistry* finish)
{
	hkSerializeUtil::ErrorDetails err;
	hkResource* resource = hkSerializeUtil::load(streamIn, &err, hkSerializeUtil::LoadOptions().useTypeInfoRegistry(finish) );

	if( resource == HK_NULL )
	{
		HK_WARN( 0x3bf82a57, err.defaultMessage.cString() );
		return HK_NULL;
	}

	void* contents = resource->getContentsPointer( expectedClass.getName(), finish );
	if( contents != HK_NULL )
	{
		m_loadedData.pushBack(static_cast<hkPackfileData*>(resource));
	}

	return contents;
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
