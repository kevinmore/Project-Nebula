/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Memory/Allocator/Pooled/hkPooledAllocator.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>

#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Serialize/Data/Util/hkDataObjectUtil.h>
#include <Common/Serialize/Packfile/Binary/hkBinaryPackfileWriter.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileCommon.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileReader.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileWriter.h>
#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileReader.h>
#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileWriter.h>
#include <Common/Serialize/Util/hkNativePackfileUtils.h>
#include <Common/Serialize/Util/hkSerializeDeprecated.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Serialize/Version/hkVersionUtil.h>

//#include <Common/Base/Monitor/hkMonitorStream.h>

#define RAISE_ERROR(err, id, msg) if( err ) err->raiseError( id, msg )

namespace
{
	static hkBool HK_CALL classSignaturesUpToDate(const hkPackfileHeader *packfileHeader, const hkClassNameRegistry* classReg);

	static const char loadVersion0FailedBoilerPlate[] = 
		"No information about this class exists in the serialization system. "
		"This could happen if:\n"
		"* You have provided a custom classes file which does not include this class\n"
		"* You have not specified keycodes before including hkProductFeatures.cxx\n"
		"* Custom classes have not been registered.";

	static const char versioningFailedBoilerPlate[] = 
		"Patching to latest version failed. Have you registered the necessary patches? " \
		"Patches may be missing because your assets are older than HK_SERIALIZE_MIN_COMPATIBLE_VERSION (if defined). " \
		"Make sure that the patches are registered properly (see hkRegisterPatches.cxx, included by hkProductFeatures.cxx). "
		"See the hkError output for more details.";

	static const char loadingFailedBoilerPlate[] = 
		"Loading tagfile failed. Is it from a newer version of the SDK?  "\
		"See the hkError output for more details";

	static hkBool32 checkTagfileWorld( hkDataWorld& world, hkSerializeUtil::ErrorDetails& errorOut, const hkVersionPatchManager& versionManager, const hkClassNameRegistry& classReg )
	{
		hkArray<hkDataClassImpl*>::Temp classes;
		world.findAllClasses(classes);
		for( int i = 0; i < classes.getSize(); ++i )
		{
			const char* name = classes[i]->getName();
			int version = classes[i]->getVersion();
			const hkClass* k = classReg.getClassByName(name);
			if( !k || k->getDescribedVersion() != version )
			{
				hkUint64 uid = versionManager.getUid(name, version);
				if( versionManager.getPatchIndex(uid) == -1 )
				{
					hkStringBuf sb;
					if( (hkUlong)version == 0 )
					{
						sb.printf("Unable to load class %s, version 0x0\n%s", name, loadVersion0FailedBoilerPlate);
					}
					else
					{
						sb.printf("Unable to version data of class %s, version 0x%p\n%s", name, (void*)(hkUlong)version, versioningFailedBoilerPlate);
					}
					errorOut.raiseError( hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, sb.cString() );
					HK_WARN(0x371d81e7, sb.cString());
					return false;
				}
			}
		}
		return true;
	}

	static hkResult loadBinaryTagfileIntoWorld( hkStreamReader& streamIn, hkDataWorldDict& world, hkSerializeUtil::ErrorDetails* errorOut, const hkClassNameRegistry& classReg )
	{
		//HK_TIMER_BEGIN("hkBinaryTagfileReader::load", HK_NULL);
		hkBinaryTagfileReader reader;
		hkDataObject obj = reader.load( &streamIn, world );
		//HK_TIMER_END();

		if ( obj.isNull() )
		{
			RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, versioningFailedBoilerPlate);
			return HK_FAILURE;
		}

		// If your patches are out of date, then applyPatches will fail.
		// This is a quick check whether applyPatches will fail, and tries to give a more meaningful error than "versioning failed"/
		if( errorOut && !checkTagfileWorld(world, *errorOut, hkVersionPatchManager::getInstance(), classReg) )
		{
			// errorOut set by checkTagfileWorld
			return HK_FAILURE;
		}

		hkDefaultClassWrapper wrapper(&classReg);

		//HK_TIMER_BEGIN("hkVersionPatchManager::applyPatches", HK_NULL);
		hkResult res = hkVersionPatchManager::getInstance().applyPatches(world, &wrapper);
		//HK_TIMER_END();

		if( res != HK_SUCCESS )
		{
			RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, versioningFailedBoilerPlate);
			return HK_FAILURE;
		}

		return HK_SUCCESS;
	}

	static hkResult loadXmlTagfileIntoWorld( hkStreamReader& streamIn, hkDataWorldDict& world, hkSerializeUtil::ErrorDetails* errorOut, const hkClassNameRegistry& classReg )
	{
		hkXmlTagfileReader reader;
		hkDataObject obj = reader.load( &streamIn, world );

		if ( obj.isNull() )
		{
			RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, versioningFailedBoilerPlate);
			return HK_FAILURE;
		}

		// If your patches are out of date, then applyPatches will fail.
		// This is a quick check whether applyPatches will fail, and tries to give a more meaningful error than "versioning failed"/
		if( errorOut && !checkTagfileWorld(world, *errorOut, hkVersionPatchManager::getInstance(), classReg) )
		{
			return HK_FAILURE;
		}
		hkDefaultClassWrapper wrapper(&classReg);
		if( hkVersionPatchManager::getInstance().applyPatches(world, &wrapper) != HK_SUCCESS )
		{
			RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, versioningFailedBoilerPlate);
			return HK_FAILURE;
		}
		return HK_SUCCESS;
	}

	
	static hkResource* HK_CALL loadPackfile
		( hkStreamReader& streamIn
		, const hkSerializeUtil::FormatDetails& details
		, hkSerializeUtil::ErrorDetails* errorOut
		, const hkTypeInfoRegistry& typeReg
        , hkSerializeUtil::LoadOptions options )
	{
		hkArray<char>::Temp buffer;
		if( details.m_formatType == hkSerializeUtil::FORMAT_PACKFILE_BINARY )
		{
			if( hkString::memCmp( &details.m_layoutRules, &hkStructureLayout::HostLayoutRules, sizeof(hkStructureLayout::HostLayoutRules) ) == 0 )
			{
				if( details.m_version == hkVersionUtil::getCurrentVersion() && 
					details.m_formatVersion == static_cast<int>(hkBinaryPackfileWriter::CURRENT_FILE_VERSION) )
				{
					// copy stream into buffer. We've already special cased the buffer version
					// of ::load, so we're not doing a redundant copy here
					
					//read in the packfile header
					buffer.expandBy(sizeof(hkPackfileHeader));
					streamIn.read(&buffer[0], sizeof(hkPackfileHeader));

					int sectionHeaderSize = 0;
					int numSections = 0;
					int offset = 0;
					{
						//read info out of the header before we expand the buffer (as that will invalidate any pointers we hold on to)
						hkPackfileHeader* header = reinterpret_cast<hkPackfileHeader*>(&buffer[0]);
						sectionHeaderSize = header->m_numSections * sizeof(hkPackfileSectionHeader);
						numSections = header->m_numSections;
						offset = static_cast<int>(hkGetByteOffset(&buffer[0], header->getSectionHeader(&buffer[0], 0)));
					}                    
                    
					int remainingSize = 0;
					buffer.expandBy(sectionHeaderSize);
					{
						hkPackfileSectionHeader* sections = reinterpret_cast<hkPackfileSectionHeader*>(&buffer[offset]);                    
						streamIn.read(sections, sectionHeaderSize);                    
					
						//based on packfile section headers, calculate total size of the packfile
						for( int i = 0; i < numSections; ++i )
						{
							remainingSize += sections[i].m_endOffset;
						}
					}
					offset += sectionHeaderSize;
					
                    
                    //read in the rest of the packfile
                    buffer.reserveExactly(buffer.getSize() + remainingSize);
                    buffer.expandBy(remainingSize);
                    streamIn.read(&buffer[offset],remainingSize);

                    if(classSignaturesUpToDate(reinterpret_cast<hkPackfileHeader *>(buffer.begin()), options.getClassNameRegistry() ) )
				    {
					    return hkNativePackfileUtils::load(buffer.begin(), buffer.getSize(), &typeReg );
				    }
					else if( options.anyIsSet( hkSerializeUtil::LOAD_FAIL_IF_VERSIONING ) )
					{
						RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, "Class signatures not up to date.");
						return HK_NULL;
					}
					
				}
				// else old binary packfile, fallthrough to deprecated
			}
			else
			{
				RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_PACKFILE_PLATFORM, "Wrong platform for packfile");
				return HK_NULL;
			}
		}

		// we only support xml and old binaries if deprecated lib is initialized

		if(buffer.getSize())
		{
			// Buffer is already loaded so don't load it again
			hkIstream bufferStream(buffer.begin(), buffer.getSize());
			return hkSerializeDeprecated::getInstance().loadOldPackfile(*bufferStream.getStreamReader(), details, errorOut);
		}
		else
		{
			return hkSerializeDeprecated::getInstance().loadOldPackfile(streamIn, details, errorOut);
		}
	}

		// This listener wraps the packfile listeners so we can reuse their code (storage shape conversion etc)
	class ForwardingPackfileListerer : public hkTagfileWriter::AddDataObjectListener
	{
		public:
			
			ForwardingPackfileListerer(hkPackfileWriter::AddObjectListener *listener, hkDataWorldNative& world, hkClassNameRegistry* classReg)
				: m_listener(listener), m_nativeWorld(world), m_classReg(classReg)
			{ }

			virtual hkDataObject addDataObjectCallback(const hkDataObject& object)
			{
				if( m_listener && !object.isNull() )
				{
					hkDataClass dataClass = object.getClass();
					const hkClass* classFromRegistry = m_classReg->getClassByName(dataClass.getName());
					// make sure that the class matches runtime version, so the callback function can safely call object functions
					HK_ASSERT(0x7cf7a5d5, classFromRegistry->getDescribedVersion() == dataClass.getVersion());
					hkDataObject::Handle h = object.getHandle();
					const void* objPtr = h.p0;
					HK_ASSERT(0x18a97b2c, objPtr);
					const hkClass* originalClass = h.p1 ? static_cast<hkClass*>(h.p1) : classFromRegistry;
					const hkClass* classPtr = originalClass;
					HK_ASSERT(0x291e0e54, classPtr);
					m_listener->addObjectCallback(objPtr, classPtr);
					if( objPtr != h.p0 || classPtr != originalClass )
					{
						return wrapObject(const_cast<void*>(objPtr), *classPtr);
					}
				}
				return object;
			}

		private:

			hkPackfileWriter::AddObjectListener* m_listener;
			hkDataWorldNative& m_nativeWorld;
			hkClassNameRegistry* m_classReg;

			inline hkDataObject wrapObject(void* object, const hkClass& klass) const
			{
				if( object )
				{
					return m_nativeWorld.wrapObject(object, klass);
				}
				return hkDataObject(HK_NULL);
			}
	};

	static hkObjectResource* loadNewPackfileOnHeap(
		hkStreamReader* streamIn,
		hkSerializeUtil::FormatDetails& details,
		hkSerializeUtil::ErrorDetails* errorOut,
		const hkClassNameRegistry* classReg,
		const hkTypeInfoRegistry* typeReg,
		hkSerializeUtil::LoadOptions options )
	{
		// When calling the finish function, set the flag to 0.
		// The normal registry sets the flag to 1 which means code is run in the finish ctors.
		// This is not what we want since the objects we have are only temp ones which are going
		// to be copied onto the heap
		// 
		// Enable memory trackers - By passing 1 as second parameter.
		hkTypeInfoRegistry noFinish(0, 1); noFinish.merge(*typeReg);
		hkRefPtr<hkResource> res; res.setAndDontIncrementRefCount( loadPackfile(*streamIn, details, errorOut, noFinish, options) );
		if ( res != HK_NULL )
		{
			hkDataWorldNative world;
			world.setClassRegistry(classReg);
			const hkClass* contentsClass = classReg->getClassByName( res->getContentsTypeName() );
			if( contentsClass )
			{
				hkDataObject contents( world.wrapObject( res->getContentsPointer(HK_NULL, typeReg), *contentsClass ) );
				return hkDataObjectUtil::toObjectWithRegistry(contents, classReg, typeReg, true);
			}
		}
		return HK_NULL;
	}

	static hkBool HK_CALL classSignaturesUpToDate(const hkPackfileHeader *packfileHeader, const hkClassNameRegistry* classReg)
	{
		const void *packfileData = static_cast<const void *>(packfileHeader);

		HK_ASSERT2(0x2a58227a, classReg, "Invalid class name registry");
		HK_ASSERT2(0x1efc54ce, packfileHeader->m_numSections, "Packfile has no sections");

		hkPackfileSectionHeader* inSection;
		int section;
		// See which section contains object name list (usually section 0)
		for(section=0, inSection = (hkPackfileSectionHeader*)(packfileHeader + 1); section < packfileHeader->m_numSections; section++, inSection++)
		{
			if(!hkString::strCmp(inSection->m_sectionTag, hkBinaryPackfileWriter::SECTION_TAG_CLASSNAMES))
			{

				// offset points to the start of the signature + name
				int offset = 0;
				// The data struct is padded out to 16 bytes. Anything with less than 6 bytes
				// must be padding, 6 or more will be detected as the name is 0xff
				while(offset < (inSection->getDataSize() - 6))
				{
					// A 4-byte signature, one-byte separator and then the filename. dstOff points to the start of the filename
					typedef struct { hkUint8 sig[4]; char sep; char name; } tempStructType;

					const tempStructType *tempStruct = reinterpret_cast<const tempStructType *>(hkAddByteOffsetConst(packfileData, inSection->m_absoluteDataStart + offset));

					//hkUint32 objSignature = tempStruct->sig;
					// The signature is not aligned so we can't use a simple hkUint32 assignment
					hkUint32 objSignature = 0;
					for(int i=0; i < hkSizeOf(hkUint32); i++)
					{
						((hkUint8*)&objSignature)[i] = tempStruct->sig[i];
					}

					const signed char *localClassName = (signed char*)&(tempStruct->name);

					if(localClassName[0] == (signed char)(-1) )
					{
						// Assume anything with 0xff as the name is padding
						break;
					}

					if(tempStruct->sep != '\t')
					{
						// Unexpected format layout, fall back to the old loader
						return false;
					}

					const hkClass *localClass = classReg->getClassByName((const char*)localClassName);
					if(!localClass)
					{
						return false;
					}
					hkUint32 localSignature = localClass->getSignature();

					if(objSignature != localSignature)
					{
						return false;
					}
					// 5 byte signature struct, length of string plus null byte
					offset += (5 + hkString::strLen((const char*)&(tempStruct->name)) + 1);
				}
				return true;
			}
		}
		// If we don't find a contents section, fall back
		return false;
	}
}

const hkClassNameRegistry* hkSerializeUtil::LoadOptions::getClassNameRegistry() const
{
	return m_classNameReg != HK_NULL
		? m_classNameReg
		: hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
}

const hkTypeInfoRegistry* hkSerializeUtil::LoadOptions::getTypeInfoRegistry() const
{
	return m_typeInfoReg != HK_NULL
		? m_typeInfoReg
		: hkBuiltinTypeRegistry::getInstance().getTypeInfoRegistry();
}

//
// Load into resource
//

hkResource* HK_CALL hkSerializeUtil::load( hkStreamReader* streamIn, ErrorDetails* errorOut, LoadOptions options )
{
	if( streamIn == HK_NULL )
	{
		RAISE_ERROR( errorOut, ErrorDetails::ERRORID_READ_FAILED, "Stream pointer is null");
		return HK_NULL;
	}

	if ( !streamIn->isOk() )
	{
		RAISE_ERROR( errorOut, ErrorDetails::ERRORID_READ_FAILED, "Stream is not ok");
		return HK_NULL;
	}

	const hkClassNameRegistry* classReg = options.getClassNameRegistry();
	const hkTypeInfoRegistry* typeInfoReg = options.getTypeInfoRegistry();

	hkSerializeUtil::FormatDetails details; hkSerializeUtil::detectFormat( streamIn, details, errorOut );
	switch( details.m_formatType )
	{
		case FORMAT_PACKFILE_BINARY:
		case FORMAT_PACKFILE_XML:
		{
			return loadPackfile( *streamIn, details, errorOut, *typeInfoReg, options );
		}
		case FORMAT_TAGFILE_BINARY:
		{
			const int tempArraySize = 16384;
			hkLocalBuffer<char> tempArray(tempArraySize);
			hkMemoryAllocator& heap = hkMemoryRouter::getInstance().heap();
			//hkPooledAllocator memPool; memPool.init(&heap, &heap, &heap, 16*1024, tempArray.begin(), tempArraySize);
			hkMemoryAllocator& memPool = heap;
			hkResource* res = HK_NULL;
			{
				hkDataWorldDict world(&memPool);				
				if( loadBinaryTagfileIntoWorld( *streamIn, world, errorOut, *classReg ) == HK_SUCCESS || options.allAreSet(LOAD_FORCED) )
				{
					//HK_TIMER_BEGIN("hkDataObjectUtil::toResourceWithRegistry", HK_NULL);
					res = hkDataObjectUtil::toResourceWithRegistry(world.getContents(), classReg, true);
					//res = hkDataObjectUtil::toResourceWithRegistry(world.getContents(), classReg, false);
					//HK_TIMER_END();
				}
			}
			//memPool.quit();
			return res;
		}
		case FORMAT_TAGFILE_XML:
		{
			const int tempArraySize = 16384;
			hkLocalBuffer<char> tempArray(tempArraySize);
			hkMemoryAllocator& heap = hkMemoryRouter::getInstance().heap();
			//hkPooledAllocator memPool; memPool.init(&heap, &heap, &heap, 16*1024, tempArray.begin(), tempArraySize);
			hkMemoryAllocator& memPool = heap;
			hkResource* res = HK_NULL;
			{
				hkDataWorldDict world(&memPool);
				if( loadXmlTagfileIntoWorld( *streamIn, world, errorOut, *classReg ) == HK_SUCCESS || options.allAreSet(LOAD_FORCED)  )
				{
					res = hkDataObjectUtil::toResourceWithRegistry(world.getContents(), classReg, true);
				}
			}
			//memPool.quit();
			return res;
		}
		case FORMAT_UNKNOWN:
		case FORMAT_ERROR:
		{
			break;
		}
	}
	RAISE_ERROR( errorOut, ErrorDetails::ERRORID_UNSUPPORTED_FORMAT, "Unable to detect format from stream");
	if( errorOut )
	{
		HK_WARN(0x5ef4a323, errorOut->defaultMessage.cString());
	}
	return HK_NULL;	
}

hkResource* HK_CALL hkSerializeUtil::load( const char* filename, ErrorDetails* resultOut, LoadOptions options )
{
	return load( hkIstream(filename).getStreamReader(), resultOut, options );
}



hkResource* HK_CALL hkSerializeUtil::load( const void* buf, int buflen, ErrorDetails* resultOut, LoadOptions options )
{
	hkPackfileHeader packfileActual; // the default magic values.
	const hkPackfileHeader* packfileLoaded = reinterpret_cast<const hkPackfileHeader*>(buf);
	//Optimisation now reenabled using class signatures and not unreliable version information
	if(1)
	{
		if((packfileLoaded->m_magic[0] == packfileActual.m_magic[0]) && (packfileLoaded->m_magic[1] == packfileActual.m_magic[1])
			&& hkString::memCmp( &packfileLoaded->m_layoutRules, &hkStructureLayout::HostLayoutRules, sizeof(hkStructureLayout::HostLayoutRules)) == 0
			&& hkString::strCmp( packfileLoaded->m_contentsVersion, hkVersionUtil::getCurrentVersion() ) == 0 &&
			classSignaturesUpToDate(packfileLoaded, options.getClassNameRegistry()))
		{
			return hkNativePackfileUtils::load(buf, buflen, HK_NULL ); // special case if we have an up to date packfile
		}
	}
	return load( hkIstream(buf, buflen).getStreamReader(), resultOut, options );
}

//
// Load into heap objects
//
hkObjectResource* HK_CALL hkSerializeUtil::loadOnHeap( hkStreamReader* streamIn, ErrorDetails* errorOut, LoadOptions options )
{
	if( streamIn == HK_NULL )
	{
		RAISE_ERROR( errorOut, ErrorDetails::ERRORID_READ_FAILED, "Stream pointer is null");
		return HK_NULL;
	}

	if ( !streamIn->isOk() )
	{
		RAISE_ERROR( errorOut, ErrorDetails::ERRORID_READ_FAILED, "Stream is not ok");
		return HK_NULL;
	}

	const hkClassNameRegistry* classReg = options.getClassNameRegistry();
	const hkTypeInfoRegistry* typeReg = options.getTypeInfoRegistry();

	hkSerializeUtil::FormatDetails details; hkSerializeUtil::detectFormat( streamIn, details, errorOut );
	switch( details.m_formatType )
	{
		case FORMAT_PACKFILE_BINARY:
		{
			if( details.m_layoutRules == hkStructureLayout::HostLayoutRules && 
				details.m_version == HAVOK_SDK_VERSION_STRING &&
				details.m_formatVersion == static_cast<int>(hkBinaryPackfileWriter::CURRENT_FILE_VERSION) )
			{
				return loadNewPackfileOnHeap(streamIn, details, errorOut, classReg, typeReg, options);
			}

			// Don't fall through to deprecated loading if we don't want to allow versioning
			if( options.anyIsSet( hkSerializeUtil::LOAD_FAIL_IF_VERSIONING ) )
			{
				RAISE_ERROR( errorOut, ErrorDetails::ERRORID_VERSIONING_FAILED, "Packfile required versioning but versioning not supported in this context.");
				return HK_NULL;
			}

			// fall through to deprecated loading
		}
		case FORMAT_PACKFILE_XML:
		{
			return hkSerializeDeprecated::getInstance().loadOldPackfileOnHeap(*streamIn, details, errorOut );
		}
		case FORMAT_TAGFILE_BINARY:
		{
			hkDataWorldDict world;
			if( loadBinaryTagfileIntoWorld( *streamIn, world, errorOut, *classReg ) == HK_SUCCESS )
			{
				return hkDataObjectUtil::toObjectWithRegistry(world.getContents(), classReg, typeReg, true);
			}
			return HK_NULL;
		}
		case FORMAT_TAGFILE_XML:
		{
			hkDataWorldDict world;
			if( loadXmlTagfileIntoWorld( *streamIn, world, errorOut, *classReg ) == HK_SUCCESS )
			{
				return hkDataObjectUtil::toObjectWithRegistry(world.getContents(), classReg, typeReg, true);
			}
			return HK_NULL;
		}
		case FORMAT_UNKNOWN:
		case FORMAT_ERROR:
		{
			break;
		}
	}
	RAISE_ERROR( errorOut, ErrorDetails::ERRORID_UNSUPPORTED_FORMAT, "Unable to detect format from stream");
	if( errorOut )
	{
		HK_WARN(0x5ef4a323, errorOut->defaultMessage.cString());
	}
	return HK_NULL;	
}

hkObjectResource* HK_CALL hkSerializeUtil::loadOnHeap( const char* filename, ErrorDetails* errorOut, LoadOptions options )
{
	return loadOnHeap( hkIstream(filename).getStreamReader(), errorOut, options );
}

hkObjectResource* HK_CALL hkSerializeUtil::loadOnHeap( const void* buf, int buflen, ErrorDetails* errorOut, LoadOptions options )
{
	return loadOnHeap( hkIstream(buf, buflen).getStreamReader(), errorOut, options );
}

hkResult HK_CALL hkSerializeUtil::savePackfile( const void* object, const hkClass& klass, hkStreamWriter* writer, const hkPackfileWriter::Options& packFileOptions, hkPackfileWriter::AddObjectListener* userListener, SaveOptions options )
{
	if( writer )
	{
		// Since the other one is const we create a copy
		hkPackfileWriter::Options localPackFileOptions(packFileOptions);
		if( options.get(SAVE_SERIALIZE_IGNORED_MEMBERS) )
		{
			localPackFileOptions.m_writeSerializedFalse = hkBool(true);
		}
		if( options.get(SAVE_TEXT_FORMAT) == 0 )
		{
			hkBinaryPackfileWriter pw(localPackFileOptions);
			pw.setContents(object, klass, userListener);
			return pw.save( writer, localPackFileOptions );
		}
		hkSerializeUtil::ErrorDetails error;
		hkResult res = hkSerializeDeprecated::getInstance().saveXmlPackfile(object, klass, writer, localPackFileOptions, userListener, &error );
		if( res == HK_FAILURE )
		{
			HK_WARN_ALWAYS(0x1d25e54f, error.defaultMessage.cString());
		}
		return res;
	}

	return HK_FAILURE;
}

hkResult HK_CALL hkSerializeUtil::saveTagfile( const void* object, const hkClass& klass, hkStreamWriter* writer, hkPackfileWriter::AddObjectListener* userListener, SaveOptions options )
{
	hkDataWorldNative world(options.get(SAVE_SERIALIZE_IGNORED_MEMBERS) !=0 );
	world.setContents(const_cast<void*>(object), klass);

	// if we ever need to pass a custom registry to this method, add it to the SaveOptions structure like LoadOptions
	ForwardingPackfileListerer forwardingListener(userListener, world, hkBuiltinTypeRegistry::getInstance().getClassNameRegistry() );

	if( writer )
	{		
		hkTagfileWriter::Options tagfileOptions;

		// Setup tagfile options based on the options passed in
		if (!options.anyIsSet(hkSerializeUtil::SAVE_CONCISE))
		{
			tagfileOptions.useVerbose(true);
		}
		if (!options.anyIsSet(hkSerializeUtil::SAVE_TEXT_NUMBERS))
		{
			tagfileOptions.useExact(true);
		}

		if( options.get(hkSerializeUtil::SAVE_TEXT_FORMAT) )
		{
			return hkXmlTagfileWriter().save(world.getContents(), writer, &forwardingListener, tagfileOptions);
		}
		else
		{
			return hkBinaryTagfileWriter().save(world.getContents(), writer, &forwardingListener, tagfileOptions);
		}
	}
	return HK_FAILURE;
}

hkResult HK_CALL hkSerializeUtil::save( const void* object, const hkClass& klass, hkStreamWriter* writer, SaveOptions options )
{
	return saveTagfile(object, klass, writer, HK_NULL, options);
}

hkBool32 HK_CALL hkSerializeUtil::isLoadable(hkStreamReader* sr)
{
	if (!sr || !sr->isOk())
	{
		return false;
	}

	if( hkTagfileReader::detectFormat( sr ) > hkTagfileReader::FORMAT_UNKNOWN )
	{
		return true;
	}
	hkSerializeUtil::FormatDetails details;
	hkSerializeUtil::detectFormat( sr, details );

	hkBool platformBinary = details.m_formatType == hkSerializeUtil::FORMAT_PACKFILE_BINARY 
		&& hkString::memCmp( &details.m_layoutRules, &hkStructureLayout::HostLayoutRules, sizeof(hkStructureLayout::HostLayoutRules)) == 0;

	// no intermediate files and host platform only
	if( platformBinary && 
		details.m_version == hkVersionUtil::getCurrentVersion() &&
		details.m_formatVersion == static_cast<int>(hkBinaryPackfileWriter::CURRENT_FILE_VERSION) )
	{
		return true;
	}
	else if( (platformBinary || details.m_formatType == hkSerializeUtil::FORMAT_PACKFILE_XML)
		&& hkSerializeDeprecated::getInstance().isLoadable(details) ) // only support xml and old binaries if deprecated lib is initialized
	{
		return true;
	}
	return false;
}

hkEnum<hkSerializeUtil::FormatType,hkInt32> HK_CALL hkSerializeUtil::detectFormat( hkStreamReader* reader, ErrorDetails* errorOut )
{
	FormatDetails details;
	detectFormat(reader, details, errorOut);
	return details.m_formatType;
}

static const char* skipSpaces(const char* text)
{
	while( *text && *text == ' ' )
	{
		++text;
	}
	return text;
}

static hkBool32 findTextValue(const char* text, const char* name, hkStringPtr& valueOut)
{
	const char *valueStart = hkString::strStr(text, name);
	if( !valueStart )
	{
		return false;
	}
	valueStart += hkString::strLen(name);
	valueStart = skipSpaces(valueStart);
	if( *valueStart != '=' )
	{
		return false;
	}
	valueStart = skipSpaces(++valueStart);
	if( *valueStart != '"' )
	{
		return false;
	}
	const char* endQuote = hkString::strChr(++valueStart, '"');
	if( !endQuote )
	{
		return false;
	}
	valueOut.set(valueStart, (hkInt32)(endQuote-valueStart));
	return true;
}

namespace
{
	class PeekStreamReader : public hkStreamReader
	{
		public:
			HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_STREAM);

			PeekStreamReader(hkStreamReader* underlying)
				: m_underlyingStream(underlying), m_curPos(0)
			{
				expandPeekBuffer(512);
			}

			virtual hkBool isOk() const
			{
				return m_curPos >= 0 && m_underlyingStream->isOk();
			}

			virtual int read(void* buf, int nbytes)
			{
				const int availBytes = peek(buf, nbytes);
				m_curPos += availBytes;

				return availBytes;
			}

			virtual int peek(void* buf, int nbytes)
			{
				if(m_curPos + nbytes >= m_peekBuffer.getSize())
				{
					expandPeekBuffer(nbytes);
				}

				const int availBytes = hkMath::min2(m_peekBuffer.getSize() - m_curPos, nbytes);

				if(availBytes)
				{
					hkString::memCpy(buf, &m_peekBuffer[m_curPos], availBytes);
				}

				return availBytes;
			}

			virtual hkSeekableStreamReader* isSeekTellSupported() { return HK_NULL; }

			void resetPos() { m_curPos = 0; }

		private:

			void expandPeekBuffer(const int nbytes)
			{
				const int newSize = hkMath::max2(m_peekBuffer.getSize() + nbytes + 1, m_peekBuffer.getSize() * 1.5);
				m_peekBuffer.setSize(newSize);
				const int actualSize = m_underlyingStream->peek(&m_peekBuffer[0], newSize);
				m_peekBuffer.setSize(actualSize);
			}

			hkStreamReader* m_underlyingStream;

			hkArray<char>::Temp m_peekBuffer;
			int m_curPos;
	};
}

void HK_CALL hkSerializeUtil::detectFormat(const char* fileName, FormatDetails& details, ErrorDetails* errorOut)
{
	detectFormat(hkIstream(fileName).getStreamReader(), details, errorOut);
}

void HK_CALL hkSerializeUtil::detectFormat( hkStreamReader* reader, FormatDetails& details, ErrorDetails* errorOut )
{
	// clear details
	details.m_formatType = FORMAT_ERROR;
	details.m_formatVersion = 0;
	details.m_version = HK_NULL;
	hkMemUtil::memSet( &details.m_layoutRules, 0, sizeof(hkStructureLayout::LayoutRules) );

	// Peek at first few bytes to check if the file is binary

	PeekStreamReader peeker(reader);

	hkBinaryTagfileReader tagFileReader;
	hkBinaryTagfile::Header tagFileHeader;
	if(tagFileReader.readHeader(&peeker, tagFileHeader) == HK_SUCCESS)
	{
		details.m_formatType = FORMAT_TAGFILE_BINARY;
		return;
	}
	
	peeker.resetPos();
	hkPackfileHeader packfileHeader;
	if(hkPackfileHeader::readHeader(&peeker, packfileHeader) == HK_SUCCESS)
	{
		details.m_formatType = FORMAT_PACKFILE_BINARY;
		details.m_version = packfileHeader.m_contentsVersion;
		details.m_formatVersion = packfileHeader.m_fileVersion;
		
		hkMemUtil::memCpy( &details.m_layoutRules, &packfileHeader.m_layoutRules, sizeof(hkStructureLayout::LayoutRules) );
		return;
	}

	peeker.resetPos();
	hkXmlTagfileReader xmlTagfileReader;
	hkXmlTagfile::Header xmlTagfileHeader;
	if (xmlTagfileReader.readHeader(&peeker, xmlTagfileHeader) == HK_SUCCESS)
	{
		details.m_formatType = FORMAT_TAGFILE_XML;
		return;
	}

	// Leaving xml packfiles for last.
	// readXmlPackfileHeader() may fail not because the file isn't an xml packfile, but just because 
	// HK_EXCLUDE_FEATURE_SerializeDeprecatedPre700 is defined.
	peeker.resetPos();
	hkSerializeDeprecated::XmlPackfileHeader xmlPackfileHeader;
	if(hkSerializeDeprecated::getInstance().readXmlPackfileHeader(&peeker, xmlPackfileHeader, errorOut) == HK_SUCCESS)
	{
		details.m_formatType = FORMAT_PACKFILE_XML;
		details.m_version = xmlPackfileHeader.m_contentsVersion;
		details.m_formatVersion = xmlPackfileHeader.m_classVersion;
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
