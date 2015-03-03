/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/Serialize/hkObjectSerialize.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileWriter.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Process/hkRemoteObjectProcess.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>

#define CHECK_STREAM(S, V) { if (!S->isOk()) return V; }

hkUint64 hkObjectSerialize::readObject( hkStreamReader* s, void*& data, hkUint32& dataSize, hkArray<GlobalFixup>& globalFixups, hkUint64& klassID)
{
	// The whole packet is assumed to have been targeted at this platform
	// so we don't need to endian swap etc.

	// id
	hkUint64 id;
	s->read(&id, sizeof(id));
	CHECK_STREAM(s,0)

		// read class ID, as it may be a virtual class we asked for, and here we know its exact, instance class type.
	s->read(&klassID, sizeof(klassID));
	CHECK_STREAM(s,0)

	// local fixups	
	hkArray<LocalFixup> localFixups;
	hkUint32 numLocal;
	s->read(&numLocal, sizeof(hkUint32));
	CHECK_STREAM(s,0)
	
	localFixups.setSize(numLocal);
	{
		for(hkUint32 li=0; (li < numLocal); ++li)
		{
			LocalFixup& lf = localFixups[li];
			s->read(&lf.fromOffset, sizeof(hkInt32));
			s->read(&lf.toOffset, sizeof(hkInt32));
			CHECK_STREAM(s,0)
		}
	}

	// global fixups
	hkUint32 numGlobal;
	s->read(&numGlobal, sizeof(hkUint32));
	CHECK_STREAM(s,0)

	globalFixups.setSize(numGlobal);
	{
		for(hkUint32 gi=0; (gi < numGlobal); ++gi)
		{
			GlobalFixup& gf = globalFixups[gi];
			s->read(&gf.fromOffset, sizeof(hkInt32));
			s->read(&gf.toID, sizeof(hkUint64));
			s->read(&gf.toClassID, sizeof(hkUint64));
			CHECK_STREAM(s,0)
		}
	}

	// data
	s->read(&dataSize, sizeof(hkUint32));
	CHECK_STREAM(s,0)

	char* dataPtr = hkAllocate<char>( dataSize, HK_MEMORY_CLASS_VDB );
	s->read(dataPtr, dataSize);
	if (!s->isOk()) 
	{
		hkDeallocate<char>( reinterpret_cast<char*>(dataPtr) );
		return 0;
	}

	// do the local fixups
	{
		for(hkUint32 li=0; (li < numLocal); ++li)
		{
			int srcOff = localFixups[li].fromOffset;
			if (srcOff < 0) continue;
			int dstOff = localFixups[li].toOffset;
			*(hkUlong*)(dataPtr+srcOff) = hkUlong(dataPtr+dstOff);
		}	
	}

	data = dataPtr;

	// done
	return id;
}

static int _writeData( hkStreamWriter* s, hkUint64 dataID, hkUint64 classID, void* data, hkUint32 dataSize, const hkRelocationInfo& reloc, hkBool32 endianSwap)
{
	HK_ASSERT2( 0x6ce419a2, dataID && data && (dataSize > 0), "Invalid data packet to write (has no size)");
	
	hkOArchive a(s, hkBool( bool(endianSwap) ));

	int bytesWritten = 0;

	// id
	a.write64u(dataID);
	CHECK_STREAM(s,-1)
	bytesWritten += 8;

	// id
	a.write64u(classID);
	CHECK_STREAM(s,-1)
	bytesWritten += 8;


	// local fixups will be performed on read on target
	hkUint32 numLocal = reloc.m_local.getSize();	
	a.write32u(numLocal);
	bytesWritten += 4;
	for (hkUint32 li=0; li < numLocal; ++li)
	{
		a.write32(reloc.m_local[li].m_fromOffset);			
		a.write32(reloc.m_local[li].m_toOffset);			
		CHECK_STREAM(s,-1)
		bytesWritten += 8;
	}
	
	// globals for convience
	hkUint32 numGlobal = reloc.m_global.getSize();	
	a.write32u(numGlobal);
	bytesWritten += 4;
	for (hkUint32  gi=0; gi < numGlobal; ++gi)
	{
		a.write32(reloc.m_global[gi].m_fromOffset);			
		a.write64u( hkUlong(reloc.m_global[gi].m_toAddress) ); //64 bit ptr (id)
		a.write64u( hkUlong(reloc.m_global[gi].m_toClass) ); // 64bit class ptr (id)
		CHECK_STREAM(s,-1)
		bytesWritten += (8*2) + 4;
	}

	// data 
	a.write32u(dataSize);
	CHECK_STREAM(s,-1)
	bytesWritten += 4;

	// the raw data chunk
	s->write(data, dataSize);
	CHECK_STREAM(s,-1)
	bytesWritten += dataSize;

	return bytesWritten;
} 

int hkObjectSerialize::writeObject( hkStreamWriter* s, const hkStructureLayout& destLayout, hkUint64 dataID, const void* data, const hkClass& klass, hkArray<GlobalFixup>& globalFixups, hkPlatformObjectWriter::Cache* cache, hkObjectCopier::ObjectCopierFlags flags )
{
	// need platform write.
	hkPlatformObjectWriter pw(destLayout, cache, flags);
	
	// The data chunk size is of unknown size until the writer has worked it out (incl array data)
	// so we write to a mem buffer and then write that to the normal stream as one big chunk.
	hkArray<char> buffer;
	hkArrayStreamWriter asw(&buffer, hkArrayStreamWriter::ARRAY_BORROW);

	// have to save the reloc info too.
	hkRelocationInfo reloc;
	pw.writeObject(&asw, data, klass, reloc );
	
	// store the global fixups.
	globalFixups.reserve( globalFixups.getSize() + reloc.m_global.getSize() );
	for (int i=0; i < reloc.m_global.getSize(); ++i)
	{
		GlobalFixup& gf = globalFixups.expandOne();
		gf.fromOffset = reloc.m_global[i].m_fromOffset;
		gf.toID = hkUlong( reloc.m_global[i].m_toAddress );
		gf.toClassID = hkUlong( reloc.m_global[i].m_toClass );
	}

	// finally write the whole thing.
	return _writeData( s, dataID, hkUlong(&klass), buffer.begin(), buffer.getSize(), reloc, destLayout.getRules().m_littleEndian != hkStructureLayout::HostLayoutRules.m_littleEndian );
}

int hkObjectSerialize::writeObject(	hkDisplaySerializeOStream* s,
									hkUchar preObjectToken,
									const void* data,
									const hkClass& klass,
									bool writePacketSize,
									bool recur,
									bool writeClass,
									const hkVtableClassRegistry* classRegistry,
									hkPlatformObjectWriter::Cache* cache,
									hkObjectCopier::ObjectCopierFlags flags)
{
	hkPointerMap<const void*, const hkClass*> writtenObjects;

	return writeObject( s, preObjectToken, data, klass, writtenObjects, writePacketSize, recur, writeClass, classRegistry, cache, flags );
}

int hkObjectSerialize::writeObject(	hkDisplaySerializeOStream* outStream,
									hkUchar preObjectToken,
									const void* data,
									const hkClass& klass,
									hkPointerMap<const void*, const hkClass*>& writtenObjects,
									bool writePacketSize,
									bool recur,
									bool writeClass,
									const hkVtableClassRegistry* classRegistry,
									hkPlatformObjectWriter::Cache* cache,
									hkObjectCopier::ObjectCopierFlags flags)
{
	hkStructureLayout pcLayout( hkStructureLayout::MsvcWin32LayoutRules );
	hkArray<hkObjectSerialize::GlobalFixup> refs;

	if ( classRegistry == HK_NULL )
	{
		classRegistry = &hkVtableClassRegistry::getInstance();
	}

	const hkClass* derivedClass = &klass;
	if ( (classRegistry != HK_NULL ) && klass.hasVtable() )
	{
		derivedClass = classRegistry->getClassFromVirtualInstance( data );
		if (!derivedClass) // revert to known abstract class..
		{
			derivedClass = &klass;
		}
	}

	// check to see if the class has already been written
	// we have to be careful not to recur when writing hkClassClass
	if ( writeClass && !writtenObjects.hasKey( derivedClass ) && ( data != derivedClass ) )
	{
		// If we have not written this class yet then recursively write it.
		writeObject( outStream, preObjectToken, derivedClass, hkClassClass, writtenObjects, writePacketSize, true, true, classRegistry, cache, flags );
	}

	// check to see if the object has already been written
	if ( writtenObjects.hasKey( data ) )
	{
		return 0;
	}

	// record that the object has been written
	writtenObjects.insert( data, derivedClass );

	// write the packet
	
	hkArray<char> buffer;
	hkArrayStreamWriter asw(&buffer, hkArrayStreamWriter::ARRAY_BORROW);
	int numBytesInObject = hkObjectSerialize::writeObject( &asw, pcLayout, hkUlong(data), data, *derivedClass, refs, cache, flags);
	if ((numBytesInObject != buffer.getSize()) || (numBytesInObject < 1))
	{
		return -1;
	}

	int packetSize = 1 + numBytesInObject;

	if ( writePacketSize )
	{
		outStream->write32u(packetSize);
	}

	outStream->write8u(preObjectToken);
	outStream->writeRaw(buffer.begin(), buffer.getSize());

	// write out all of the nested objects
	if (recur)
	{
		for (int ri=0; ri < refs.getSize(); ++ri)
		{
			const hkObjectSerialize::GlobalFixup& gf = refs[ri];
			void* object = (void*)(hkUlong)( gf.toID);
			hkClass* k = (hkClass*)(hkUlong)( gf.toClassID );
			if(k)
			{
				packetSize += writeObject( outStream, preObjectToken, object, *k, writtenObjects, writePacketSize, true, writeClass, classRegistry, cache, flags );
			}
		}
	}
	
	return packetSize;
}

void HK_CALL hkObjectSerialize::writeObject( hkDisplaySerializeOStream* stream, hkReferencedObject* object, bool writePacketSize, bool writePackfile, hkStructureLayout layout, hkPlatformObjectWriter::Cache* cache, hkObjectCopier::ObjectCopierFlags flags )
{
	HK_TIMER_BEGIN("write obj", HK_NULL);

	hkArray<char> objectAsTagfile;
	hkOstream buffer(objectAsTagfile);			
	{
		void* data = object;
		const hkClass* klass = hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry()->getClassFromVirtualInstance(object);
		if (klass)
		{
			if( !writePackfile )
			{
				hkDataWorldNative world;
				world.setContents(data, *klass);
				hkBinaryTagfileWriter().save(world.getContents(), buffer.getStreamWriter(), HK_NULL);

				//hkOstream debugBuffer( "objecttest.txt" );
				//hkSerializeUtil::saveTagfile(mesh, *meshClass, debugBuffer.getStreamWriter(), false, HK_NULL);
			}
			else
			{
				hkPackfileWriter::Options options;
				{
					options.m_layout = layout;
					options.m_writeMetaInfo = false;
				}
				
				hkSerializeUtil::savePackfile(data, *klass, buffer.getStreamWriter(), options, HK_NULL, hkSerializeUtil::SaveOptions());
			}
		}
		else
		{
			HK_WARN_ALWAYS(0x472133e, "Class not found");
			return;
		}
	}	

	const int packetSize = 1 + 4 + objectAsTagfile.getSize();

	if ( writePacketSize )
	{		
		stream->write32u(packetSize);
	}

	HK_MONITOR_ADD_VALUE( "bytes", float(packetSize), HK_MONITOR_TYPE_INT );

	stream->write8u(hkVisualDebuggerProtocol::HK_LIVE_OBJECT);
	stream->write32(objectAsTagfile.getSize());
	stream->writeRaw(objectAsTagfile.begin(), objectAsTagfile.getSize());

	HK_TIMER_END();
}

hkReferencedObject* HK_CALL hkObjectSerialize::readObject( hkDisplaySerializeIStream* stream, hkSerializeUtil::ErrorDetails& errorDetails )
{
	HK_TIMER_BEGIN("read obj", HK_NULL);

	// Read object from stream
	hkArray<char> objectAsTagfile;	
	{
		int tagfileSize = stream->read32();		
		objectAsTagfile.setSize(tagfileSize);
		stream->readRaw(objectAsTagfile.begin(), tagfileSize);		

		HK_MONITOR_ADD_VALUE( "bytes", float(tagfileSize), HK_MONITOR_TYPE_INT );
	}
	
	hkObjectResource* resource = hkSerializeUtil::loadOnHeap(objectAsTagfile.begin(), objectAsTagfile.getSize(), &errorDetails, hkSerializeUtil::LOAD_FAIL_IF_VERSIONING);
	
	// Get the object
	hkReferencedObject* object = HK_NULL;
	if( resource != HK_NULL )
	{
		const hkClass* klass = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry()->getClassByName(resource->getContentsTypeName());
		if( klass != HK_NULL )
		{						
			extern const hkClass hkReferencedObjectClass;
			if( hkReferencedObjectClass.isSuperClass(*klass) )
			{
				object = resource->getContents<hkReferencedObject>();
				object->addReference();				
			}
			else
			{
				HK_WARN_ALWAYS(0x4fda0111, "Object is not a referenced object.  Ignoring");
			}						
		}
		else
		{
			HK_WARN_ALWAYS(0x4fda0112, "Object hkClass not found.  Ignoring.");
		}
	
		resource->removeReference();
	}
	else
	{
		HK_WARN_ALWAYS(0x4fda0113, "Object could not be read from tagfile.");
	}
		
	HK_TIMER_END();
	return object;
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
