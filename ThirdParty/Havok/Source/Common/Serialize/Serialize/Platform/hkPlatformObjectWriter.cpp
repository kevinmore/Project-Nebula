/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Serialize/Platform/hkPlatformObjectWriter.h>
#include <Common/Base/System/Io/Writer/OffsetOnly/hkOffsetOnlyStreamWriter.h>
#include <Common/Serialize/Copier/hkDeepCopier.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>

hkPlatformObjectWriter::Cache::Cache()
{
}

hkPlatformObjectWriter::Cache::~Cache()
{
	for( int i = 0; i < m_allocations.getSize(); ++i )
	{
		hkDeepCopier::freeDeepCopy( m_allocations[i] );
	}
}

hkClass* hkPlatformObjectWriter::Cache::get( const hkClass* klass, const hkStructureLayout& layout )
{
	void* ret = HK_NULL;
	if( m_platformClassFromHostClass.get(klass, &ret) == HK_FAILURE)
	{
		ret = hkDeepCopier::deepCopy( klass, hkClassClass, &m_platformClassFromHostClass );
		m_allocations.pushBack(ret);
		layout.computeMemberOffsetsInplace( *static_cast<hkClass*>(ret), m_platformClassComputed );
	}
	return static_cast<hkClass*>(ret);
}



hkPlatformObjectWriter::hkPlatformObjectWriter( const hkStructureLayout& layout, class Cache* cache, hkObjectCopier::ObjectCopierFlags flags )
{
	//XXX need to handle homogeneous arrays with differing offsets
	m_copier = new hkObjectCopier( hkStructureLayout::HostLayoutRules, layout, flags );
	// if the source and target rules are identical, we don't need a cache
	
	if(getLayout().getRules() == hkStructureLayout::HostLayoutRules)
	{
		m_cache = HK_NULL;
	}
	else
	{
		if(cache)
		{
			cache->addReference();
			m_cache = cache;
		}
		else
		{
			m_cache = new Cache;
		}
	}
}

hkPlatformObjectWriter::~hkPlatformObjectWriter()
{
	m_copier->removeReference();
	if( m_cache )
	{
		m_cache->removeReference();
	}
}

extern const class hkClass hkClassClass;

hkResult hkPlatformObjectWriter::writeObject( hkStreamWriter* writer, const void* dataIn, const hkClass& klass, hkRelocationInfo& reloc )
{
	const hkClass* classCopy = m_cache
		? m_cache->get(&klass, getLayout())
		: &klass;
	hkOffsetOnlyStreamWriter dummyWriter;
	HK_ON_DEBUG( hkStreamWriter* origWriter = writer);
	if( m_cache && hkString::strCmp(classCopy->getName(), "hkClass") == 0 )
	{
		//
		// The meta data also must be saved using structure layout for the specified platform.
		// We have to call the m_copier->copyObject(...) twice:
		// 1. Save the proper class data updated (member offsets) for specified structure layout
		// (using dummy relocation info - updated class data uses pointers
		// that are discarded later on).
		// 2. "Dummy" save the original class data updating the relocation info.
		//

		// If it is requesting a hkClass, it will have the cached hkClassClass and not the original
		// If we look up the cache again we will double rewrite the cached data
		classCopy = m_cache->get(&hkClassClass, getLayout());
		const hkClass* classData = m_cache->get(static_cast<const hkClass*>(dataIn), getLayout());

		// Set dummy writer to the actual data pointer position.
		dummyWriter.seek( writer->tell(), hkStreamWriter::STREAM_SET );

		// Save the updated class data into the real stream.
		hkRelocationInfo dummyReloc;
		m_copier->copyObject( classData, klass, writer, *classCopy, dummyReloc );

		writer = &dummyWriter;
	}
	m_copier->copyObject( dataIn, klass, writer, *classCopy, reloc );
	HK_ASSERT( 0x263de130,  writer == origWriter || writer->tell() == origWriter->tell() );
	return writer->isOk() ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkPlatformObjectWriter::writeRaw( hkStreamWriter* writer, const void* data, int dataLen )
{
	return writer->write(data, dataLen) == dataLen
		? HK_SUCCESS
		: HK_FAILURE;
}

const hkStructureLayout& hkPlatformObjectWriter::getLayout() const
{
	return m_copier->getLayoutOut();
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
