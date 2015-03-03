/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkNativePackfileUtils.h>

#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>
#include <Common/Serialize/Util/hkStructureLayout.h>
#include <Common/Serialize/Version/hkVersionUtil.h>

#if 0 && defined(HK_DEBUG)
#	include <Common/Base/Fwd/hkcstdio.h>
using namespace std;
#	define TRACE(A) A
#else
#	define TRACE(A) // nothing
#endif

namespace LOCALNAMESPACE
{
	struct TrackedObjectsArray_Element
	{
		/// Typeinfo of the class that is being tracked
		hkTypeInfo *m_typeInfo;
		/// Offset from the start of the loaded buffer of object
		/// (32-bit value)
		int m_offset;
	};
	typedef hkArray<TrackedObjectsArray_Element> TrackedObjectsArray;
	static const int COPY_LOADED_FLAG = 0xD5109142;

	struct NativePackfileUtils_Location
	{
		int m_sectionIndex;
		int m_offset;
	};

	struct InplaceDataHeader
	{
#ifndef HK_ARCH_ARM
		HK_ALIGN_REAL(int m_copy_loaded_flag);
#else
		int m_copy_loaded_flag;
#endif
		int m_contentsOffset;
		TrackedObjectsArray m_trackedObjects;
		hkArray<hkPackfileSectionHeader> m_sections;
		hkTypeInfo *m_contentsType;
#ifdef HK_ARCH_ARM
		hkUint32 m_padding[3];
#endif
	};

	struct hkNativeResource : public hkResource
	{
		hkNativeResource(hkArray<char>& buf, void* contents, const char* contentsTypeName)
			: m_contents(contents)
			, m_contentsTypeName(contentsTypeName)
		{
			m_data.swap(buf);
		}

		void callDestructors()
		{
			if( m_contents != HK_NULL )
			{
				hkNativePackfileUtils::unload( m_data.begin(), m_data.getSize() );
				m_contents = HK_NULL;
				m_contentsTypeName = HK_NULL;
			}
		}

		~hkNativeResource()
		{
			callDestructors();
		}

		virtual const char* getName() const
		{
			return "hkNativeResource";
		}
		virtual void* getContentsPointer(const char* typeName, const hkTypeInfoRegistry* ignoredBecauseAlreadyDone) const
		{
			if( typeName ) // If given an expected type
			{
				const hkClassNameRegistry* reg = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
				const hkClass* actual = reg->getClassByName(m_contentsTypeName);
				const hkClass* requested = reg->getClassByName(typeName);
				if( actual && requested )
				{
					if( requested->isSuperClass(*actual) == false )
					{
						HK_WARN(0x3f7a4853, "Asked for " << typeName << " but contains " << m_contentsTypeName << ". Returning null");
						return HK_NULL;
					}
				}
				HK_WARN_ON_DEBUG_IF(actual==HK_NULL, 0x7bc748ab, actual->getName() << " not registered, unabled to check if the contents are correct");
				HK_WARN_ON_DEBUG_IF(requested==HK_NULL, 0x7bc748ac, requested->getName() << " not registered, unabled to check if the contents are correct");
			}
			return m_contents;
		}
		virtual void getImportsExports( hkArray<Import>& impOut, hkArray<Export>& expOut ) const
		{
			hkNativePackfileUtils::getImportsExports(m_data.begin(), expOut, impOut);
		}
		virtual const char* getContentsTypeName() const 
		{
			return m_contentsTypeName;
		}

		hkArray<char> m_data;
		void* m_contents;
		const char* m_contentsTypeName;
	};

	HK_COMPILE_TIME_ASSERT( sizeof(InplaceDataHeader) % 16 == 0 );
}
using namespace LOCALNAMESPACE;

hkResource* HK_CALL hkNativePackfileUtils::load(const void* packfileData, int dataSize, const hkTypeInfoRegistry* userRegistry)
{
	hkArray<char> buf;
	buf.setSize( getRequiredBufferSize(packfileData, dataSize) );
	void* contents = load( packfileData, dataSize, buf.begin(), buf.getSize(), userRegistry);
	return new hkNativeResource(buf, contents, reinterpret_cast<InplaceDataHeader*>(buf.begin())->m_contentsType->getTypeName());
}


static inline void* _getSectionDataByIndex( void* base, 
	const hkArrayBase<hkPackfileSectionHeader*>& sections, int sectionIndex, int offset )
{
	if( sections[sectionIndex]->getDataSize() ) // not loaded
	{
		HK_ASSERT( 0x4895360e, offset >= 0 );
		HK_ASSERT( 0x4895360f, offset < sections[sectionIndex]->getDataSize() );
		return hkAddByteOffset( base, sections[sectionIndex]->m_absoluteDataStart + offset );
	}
	return HK_NULL;
}

#define VALIDATE(TEST, ERRMSG) if( (TEST)==false ) { if(errOut) *errOut = (ERRMSG); return HK_FAILURE; }
hkResult HK_CALL hkNativePackfileUtils::validatePackfileHeader( const void* packfileData, const char** errOut )
{
	VALIDATE( packfileData != HK_NULL, "Pointer is null" );
	const hkPackfileHeader* header = static_cast<const hkPackfileHeader*>(packfileData);
	hkPackfileHeader magic;
	VALIDATE( (header->m_magic[0] == magic.m_magic[0]) && (header->m_magic[1] == magic.m_magic[1]), "Missing packfile magic header. Is this from a binary file?");
	const hkStructureLayout::LayoutRules& rules = hkStructureLayout::HostLayoutRules;
	VALIDATE( header->m_layoutRules[0] == rules.m_bytesInPointer, "Trying to process a binary file with a different pointer size than this platform." );
	VALIDATE( header->m_layoutRules[1] == rules.m_littleEndian, "Trying to process a binary file with a different endian than this platform." );
	VALIDATE( header->m_layoutRules[2] == rules.m_reusePaddingOptimization, "Trying to process a binary file with a different padding optimization than this platform." );
	VALIDATE( header->m_layoutRules[3] == rules.m_emptyBaseClassOptimization, "Trying to process a binary file with a different empty base class optimization than this platform." );
	VALIDATE( (hkUlong(packfileData) & 0x3) == 0, "Packfile data source needs to be 4 byte aligned");
	VALIDATE( header->m_contentsVersion[0] != (char)( -1 ), "Packfile file format is too old" );
	VALIDATE( hkString::strCmp(header->m_contentsVersion, hkVersionUtil::getCurrentVersion()) == 0, "Packfile contents are not up to date" );

	return HK_SUCCESS;
}
#undef VALIDATE

int hkNativePackfileUtils::getRequiredBufferSize( const void* packfileData, int packfileSize )
{
	HK_ASSERT2( 0x78b7ad37, packfileSize > hkSizeOf(hkPackfileHeader), "Packfile is too small" );
	HK_ASSERT( 0x78b7ad38, validatePackfileHeader(packfileData, HK_NULL) == HK_SUCCESS );
	const hkPackfileHeader* header = static_cast<const hkPackfileHeader*>(packfileData);

	int dataSize = 0;
	int numTrackedObjects = 0;

	// calculate size of objects
	for( int i = 0; i < header->m_numSections; ++i )
	{
		const hkPackfileSectionHeader* section = header->getSectionHeader(packfileData, i);
		// add data size
		dataSize += section->getDataSize();
		// add number of tracked objects for this section (may be slight overestimate)
		numTrackedObjects += (section->getFinishSize()) / (3*hkSizeOf(hkInt32));
		// add exports+imports
		dataSize += section->getExportsSize() + section->getImportsSize();
	}

	return sizeof(InplaceDataHeader) // header
		+ (sizeof(hkPackfileSectionHeader) * header->m_numSections) // section headers
		+ dataSize // object data + import/export
		+ (numTrackedObjects * sizeof(TrackedObjectsArray_Element)); // size of the map data
}


// Copy the object parts of a packfile
static void HK_CALL _copySection(const hkPackfileSectionHeader& inSection, hkPackfileSectionHeader& outSection, int outCurOffset, const void* packfileData, void* outBuffer, int packfileSize, int outBufferSize)
{
	hkString::memCpy(outSection.m_sectionTag, inSection.m_sectionTag, sizeof(inSection.m_sectionTag));
	outSection.m_nullByte = inSection.m_nullByte;
	outSection.m_absoluteDataStart = outCurOffset;
	outSection.m_localFixupsOffset = inSection.getDataSize();
	outSection.m_globalFixupsOffset = inSection.getDataSize();
	outSection.m_virtualFixupsOffset = inSection.getDataSize();
	outSection.m_exportsOffset = inSection.getDataSize();
	outSection.m_importsOffset = inSection.getDataSize() + inSection.getExportsSize();
	outSection.m_endOffset = inSection.getDataSize() + inSection.getExportsSize() + inSection.getImportsSize();

 	HK_ASSERT2(0xff668fe7, ( (inSection.m_absoluteDataStart <= packfileSize) && ((inSection.m_absoluteDataStart + inSection.m_endOffset) <= packfileSize) ),
 		"Inplace packfile data is too small. Is it corrupt?");
 	HK_ASSERT2(0x673e19d4, ( outCurOffset + inSection.getDataSize() <= outBufferSize ), "The buffer is too small.");

	// copy section : data,exports,imports
	const char* inSectionBegin = (const char*)hkAddByteOffsetConst(packfileData, inSection.m_absoluteDataStart);
	char* outSectionBegin = (char*)hkAddByteOffset(outBuffer, outCurOffset);

	hkString::memCpy(outSectionBegin,
		inSectionBegin,
		inSection.getDataSize() );
	hkString::memCpy( outSectionBegin + outSection.m_exportsOffset,
		inSectionBegin + inSection.m_exportsOffset,
		inSection.getExportsSize() );
	hkString::memCpy( outSectionBegin + outSection.m_importsOffset,
		inSectionBegin + inSection.m_importsOffset,
		inSection.getImportsSize() );
}

static void HK_CALL _applyLocalFixups(const hkPackfileSectionHeader& inSection, hkPackfileSectionHeader& outSection, 
	int outCurOffset, const void* packfileData, void* outBuffer)
{
	const char* inSectionBegin = (const char*)hkAddByteOffsetConst(packfileData, inSection.m_absoluteDataStart);
	char* outSectionBegin = (char*)hkAddByteOffset(outBuffer, outCurOffset);

	// apply local fixups now
	const int* localFixups = reinterpret_cast<const int*>(inSectionBegin + inSection.m_localFixupsOffset);
	for( int i = 0; i < inSection.getLocalSize() / hkSizeOf(hkInt32); i+=2 )
	{
		int srcOff = localFixups[i  ];
		if( srcOff == -1 ) continue;
		HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
		int dstOff = localFixups[i+1];
		void** addrSrc = reinterpret_cast<void**>(outSectionBegin+srcOff);
		void* dst = reinterpret_cast<void*>(outSectionBegin+dstOff);
		HK_ASSERT2( 0x75936f92, *addrSrc == HK_NULL,
			"Pointer has already been patched. Corrupt file or loadInPlace called multiple times?");
		*addrSrc = dst;
	}
}

static void HK_CALL _applyGlobalFixups(const hkPackfileSectionHeader& inSection, hkPackfileSectionHeader& outSection, 
	 const void* packfileData, void* outBuffer, const hkArrayBase<hkPackfileSectionHeader*>& outSectionAddresses)
{
	char* outSectionBegin = (char*)hkAddByteOffset(outBuffer, outSection.m_absoluteDataStart );
	const int* globalFixups;
	globalFixups = reinterpret_cast<const int*>( hkAddByteOffsetConst(packfileData, inSection.m_absoluteDataStart + inSection.m_globalFixupsOffset ) );

	for( int i = 0; i < inSection.getGlobalSize() / hkSizeOf(hkInt32); i += 3 )
	{
		int srcOff = globalFixups[i  ];
		if( srcOff == -1 ) continue;
		HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
		int dstSec = globalFixups[i+1];
		int dstOff = globalFixups[i+2];

		// automatically checks for dest section loaded
		void* dstPtr = _getSectionDataByIndex(outBuffer, outSectionAddresses, dstSec, dstOff);
		*(hkUlong*)(outSectionBegin+srcOff) = hkUlong(dstPtr);
	}
}

// If map is NULL, the finish object type is written into the old virtual fixup table
// allowing the destructors to be located for inplace loaded objects. The fixup
// table will no longer be valid
static void HK_CALL _applyVirtualFixups(const hkPackfileSectionHeader& inSection, hkPackfileSectionHeader& outSection, 
	const void* packfileData, void* outBuffer, const hkArrayBase<hkPackfileSectionHeader*>& outSectionAddresses, 
	TrackedObjectsArray* map, const hkTypeInfoRegistry* finishRegistry, const hkClassNameRegistry* classNameRegistry, 
	hkArray<hkVariant>& postFinishObjects)
{
	void* outSectionBegin = hkAddByteOffset(outBuffer, outSection.m_absoluteDataStart );
	const int* virtualFixups = reinterpret_cast<const int *>( hkAddByteOffsetConst( packfileData, inSection.m_absoluteDataStart + inSection.m_virtualFixupsOffset ) );
	InplaceDataHeader* outBufferHeader = (InplaceDataHeader*)( outBuffer );
	void *contentsPointer = hkAddByteOffset(outBuffer, outBufferHeader->m_contentsOffset);

	
	for( int i = 0; i < inSection.getFinishSize() / hkSizeOf(hkInt32); i += 3 )
	{
		int srcOff = virtualFixups[i  ];
		if( srcOff == -1 ) continue;
		HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
		int dstSec = virtualFixups[i+1];
		int dstOff = virtualFixups[i+2];


		// automatically checks for dest section loaded
		void* typeName = _getSectionDataByIndex(outBuffer, outSectionAddresses, dstSec, dstOff);
		void* objAddress = hkAddByteOffset(outSectionBegin, srcOff);

		const hkTypeInfo* registeredType = finishRegistry->finishLoadedObject( objAddress, static_cast<char*>(typeName) );
		// save info to cleanup the object later on
		if (registeredType)
		{
			if(map)
			{
				TRACE(printf("+ctor\t%s at %p.\n", registeredType->getTypeName(), objAddress));
				TrackedObjectsArray_Element e;
				e.m_typeInfo = const_cast<hkTypeInfo *>(registeredType);
				e.m_offset = outSection.m_absoluteDataStart + srcOff;
				map->pushBack(e);
				if(objAddress == contentsPointer)
				{
					outBufferHeader->m_contentsType = const_cast<hkTypeInfo *>(registeredType);
				}
			}
			else
			{
				int* virtualFixupsOutput = reinterpret_cast<int*>( hkAddByteOffset( outBuffer, outSection.m_absoluteDataStart + outSection.m_virtualFixupsOffset ) );
				hkUlong tempPointer = reinterpret_cast<hkUlong>(registeredType);
#ifndef HK_ARCH_ARM // snc compiler with 03 etc does not like this
				virtualFixupsOutput[i+1] = static_cast<int>((static_cast<hkUint64>(tempPointer) & 0xFFFFFFFF00000000LL) >> 32);
				virtualFixupsOutput[i+2] = static_cast<int>((static_cast<hkUint64>(tempPointer) & 0x00000000FFFFFFFFLL));
#else
				virtualFixupsOutput[i+1] = 0;
				virtualFixupsOutput[i+2] = static_cast<int>(tempPointer);
#endif
			}

			// Keep track of any post finish function
			if( classNameRegistry )
			{
				const hkClass* klass = classNameRegistry->getClassByName( registeredType->getTypeName() );
				if( klass )
				{
					const hkVariant* attr = klass->getAttribute("hk.PostFinish");
					if( attr )
					{
						hkVariant variant;
						variant.m_class = klass;
						variant.m_object = objAddress;
						postFinishObjects.pushBack( variant );
					}
				}
			}
		}
		else
		{
			HK_WARN(0x6281a9b2, "No type info registered for class " << (char*) typeName << "");
		}
	}
}


void* HK_CALL hkNativePackfileUtils::load( const void* packfileData, int packfileSize, void* outBuffer, int outBufferSize, const hkTypeInfoRegistry* userRegistry )
{
	// after loading, the output buffer looks like
	// InplaceDataHeader
	// array data of sections
	// Data from all sections concatenated
	// Exports from all sections concatenated
	// Imports from all sections concatenated
	// Map data from m_trackedObjects
	HK_ASSERT2(0xc1e7e32b, (hkUlong(outBuffer) & (HK_REAL_ALIGNMENT-1)) == 0, "Output buffer needs to be aligned for SIMD");
	HK_ASSERT2(0x673e19d4, hkSizeOf(InplaceDataHeader) <= outBufferSize, "The buffer is too small.");
	HK_ASSERT( 0x78b7ad37, validatePackfileHeader(packfileData, HK_NULL) == HK_SUCCESS );

	const hkPackfileHeader* packfileHeader = static_cast<const hkPackfileHeader*>(packfileData);

	// init and fill buffer
	InplaceDataHeader* outBufferHeader = static_cast<InplaceDataHeader*>( outBuffer );
	outBufferHeader->m_copy_loaded_flag = COPY_LOADED_FLAG;
	int outCurOffset = sizeof(InplaceDataHeader);
	new( &outBufferHeader->m_sections ) hkArray<hkPackfileSectionHeader>(
			static_cast<hkPackfileSectionHeader*>(hkAddByteOffset(outBuffer, outCurOffset)),
			packfileHeader->m_numSections,
			packfileHeader->m_numSections );
	hkArray<hkPackfileSectionHeader>& outSections = outBufferHeader->m_sections;
	outCurOffset += sizeof( hkPackfileSectionHeader ) * packfileHeader->m_numSections;

	// Array of addresses to packfile section headers.
	hkLocalArray<hkPackfileSectionHeader*> outSectionAddresses(packfileHeader->m_numSections);
	for(int i = 0; i < outSections.getSize(); ++i)
	{
		outSectionAddresses.pushBack(&outSections[i]);
	}
	
	// process sections data and initialize with data only
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		const hkPackfileSectionHeader& inSection = *packfileHeader->getSectionHeader(packfileData, sectionIndex);
		hkPackfileSectionHeader& outSection = outSections[sectionIndex];

		_copySection(inSection, outSection, outCurOffset, packfileData, outBuffer, packfileSize, outBufferSize);

		if (sectionIndex == packfileHeader->m_contentsSectionIndex)
		{
			outBufferHeader->m_contentsOffset = outCurOffset + packfileHeader->m_contentsSectionOffset;
		}

		_applyLocalFixups(inSection, outSection, outCurOffset, packfileData, outBuffer);

		outCurOffset += outSection.m_endOffset;
	}


	// apply global fixups now to objects
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		const hkPackfileSectionHeader& inSection = *packfileHeader->getSectionHeader(packfileData, sectionIndex);
		if(inSection.getGlobalSize())
		{
			hkPackfileSectionHeader& outSection = outSections[sectionIndex];
			_applyGlobalFixups(inSection, outSection, packfileData, outBuffer, outSectionAddresses);
		}
	}

	new( &outBufferHeader->m_trackedObjects ) TrackedObjectsArray(
		reinterpret_cast<TrackedObjectsArray_Element*>(hkAddByteOffset(outBuffer, outCurOffset)),
		0, outBufferSize - outCurOffset );
	const hkTypeInfoRegistry* finishRegistry = userRegistry != HK_NULL
		? userRegistry
		: hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry();
	const hkClassNameRegistry* classRegistry = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
	// finish objects

	outBufferHeader->m_contentsType = HK_NULL;

	hkArray<hkVariant> postFinishObjects;
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		const hkPackfileSectionHeader& inSection = *packfileHeader->getSectionHeader(packfileData, sectionIndex);
		hkPackfileSectionHeader &outSection = outSections[sectionIndex];
		if(inSection.getFinishSize())
		{
			_applyVirtualFixups(inSection, outSection, packfileData, outBuffer, outSectionAddresses, 
				&outBufferHeader->m_trackedObjects, finishRegistry, classRegistry, postFinishObjects);
		}
	}
	HK_ASSERT(0x5716f93c, outBufferHeader->m_trackedObjects.begin() == reinterpret_cast<TrackedObjectsArray_Element*>(hkAddByteOffset(outBuffer, outCurOffset)));

	// Apply any post finish functions
	for( int i = 0; i < postFinishObjects.getSize(); ++i )
	{
		void* ptr = postFinishObjects[i].m_object;
		const hkClass& klass = *postFinishObjects[i].m_class;
		const hkVariant* attr = klass.getAttribute("hk.PostFinish");
		HK_ASSERT2(0x1e974825, attr && (attr->m_class == &hkPostFinishAttributeClass), "Object does not have PostFinish attribute");
		const hkPostFinishAttribute* postFinishAttr = reinterpret_cast<hkPostFinishAttribute*>(attr->m_object);
		postFinishAttr->m_postFinishFunction(ptr);
	}

	return hkAddByteOffset(outBuffer, outBufferHeader->m_contentsOffset);
}

void* HK_CALL hkNativePackfileUtils::loadInPlace(void* packfileData, int dataSize, const hkTypeInfoRegistry* userRegistry, const char** errOut)
{
	if (errOut)
	{
		*errOut = HK_NULL;
	}

	HK_ASSERT2(0xc1e7e32b, (hkUlong(packfileData) & 0xf) == 0, "Output buffer needs to be 16 byte aligned");

	// Validate the header
	if (validatePackfileHeader(packfileData, errOut) != HK_SUCCESS)
	{
		return HK_NULL;
	}

	hkPackfileHeader* packfileHeader = static_cast<hkPackfileHeader*>(packfileData);

	HK_ASSERT2(0x59e14939, !(packfileHeader->m_flags & 1), "Packfile already loaded in-place");
	if(packfileHeader->m_flags & 1)
	{
		return HK_NULL;
	}
	packfileHeader->m_flags |= 1;


	// init and fill buffer
	int outputContentsOffset = -1;

	hkLocalArray<hkPackfileSectionHeader*> inSectionAddresses(packfileHeader->m_numSections);
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		hkPackfileSectionHeader& inSection = *packfileHeader->getSectionHeader(packfileData, sectionIndex);
		inSectionAddresses.pushBack(&inSection);
	}

	// process sections data and initialize with data only
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		hkPackfileSectionHeader& inSection = *inSectionAddresses[sectionIndex];
		const int outCurOffset = inSection.m_absoluteDataStart;

		if ((hkString::strCmp(inSection.m_sectionTag, "__types__") == 0 ) && inSection.m_endOffset > 0)
		{
			HK_WARN_ONCE(0x26a8cc12, "Found meta data in packfile. Consider saving without metadata to reduce filesize");
		}
		if (sectionIndex == packfileHeader->m_contentsSectionIndex)
		{
			outputContentsOffset = outCurOffset + packfileHeader->m_contentsSectionOffset;
		}

		_applyLocalFixups(inSection, inSection, outCurOffset, packfileData, packfileData);
	}


	// apply global fixups now to objects
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		hkPackfileSectionHeader& inSection = *inSectionAddresses[sectionIndex];
		if(inSection.getGlobalSize())
		{
			_applyGlobalFixups(inSection, inSection, packfileData, packfileData, inSectionAddresses);
		}
	}

	const hkTypeInfoRegistry* finishRegistry = userRegistry != HK_NULL
		? userRegistry
		: hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry();
	const hkClassNameRegistry* classRegistry = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();

	// finish objects
	hkArray<hkVariant> postFinishObjects;
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		hkPackfileSectionHeader& inSection = *inSectionAddresses[sectionIndex];
		if(inSection.getFinishSize())
		{
			_applyVirtualFixups(inSection, inSection, packfileData, packfileData, 
				inSectionAddresses, HK_NULL, finishRegistry, classRegistry, postFinishObjects);
		}
	}
	HK_ASSERT2(0x5cfe0f96, outputContentsOffset > -1, "Contents section not found");
	
	// Apply any post finish functions
	for( int i = 0; i < postFinishObjects.getSize(); ++i )
	{
		void* ptr = postFinishObjects[i].m_object;
		const hkClass& klass = *postFinishObjects[i].m_class;
		const hkVariant* attr = klass.getAttribute("hk.PostFinish");
		HK_ASSERT2(0x1e974825, attr && (attr->m_class == &hkPostFinishAttributeClass), "Object does not have PostFinish attribute");
		const hkPostFinishAttribute* postFinishAttr = reinterpret_cast<hkPostFinishAttribute*>(attr->m_object);
		postFinishAttr->m_postFinishFunction(ptr);
	}

	return hkAddByteOffset(packfileData, outputContentsOffset);
}



const char* HK_CALL hkNativePackfileUtils::getContentsClassName(const void* packfileData, int dataSize)
{
	int bufNeeded = hkSizeOf(hkPackfileHeader);
	if( packfileData && dataSize >= bufNeeded )
	{
		const hkPackfileHeader* packfileHeader = reinterpret_cast<const hkPackfileHeader*>(packfileData);
		if( (packfileHeader->m_magic[0] == 0x57e0e057) && (packfileHeader->m_magic[1] == 0x10c0c010) ) //valid packfile buffer
		{
			bufNeeded += packfileHeader->m_numSections * hkSizeOf(hkPackfileSectionHeader);
			if( packfileHeader->m_numSections > 0 && dataSize >= bufNeeded )
			{
				const hkPackfileSectionHeader* inSections = reinterpret_cast<const hkPackfileSectionHeader*>(packfileHeader + 1);

				int nameOffset = inSections[packfileHeader->m_contentsClassNameSectionIndex].m_absoluteDataStart + packfileHeader->m_contentsClassNameSectionOffset;
				if( nameOffset <= dataSize )
				{
					return static_cast<const char*>(packfileData) + nameOffset;
				}
			}
		}
	}
	
	return HK_NULL;
}

void hkNativePackfileUtils::unload( void* buffer, int bufferSize )
{
	InplaceDataHeader* bufferHeader = reinterpret_cast<InplaceDataHeader*>( buffer );
	HK_ASSERT2(0x6c1adf70, bufferHeader->m_copy_loaded_flag == COPY_LOADED_FLAG, "Invalid packfile buffer");
	HK_ASSERT2(0x673e19d4, hkSizeOf(InplaceDataHeader) <= bufferSize, "The buffer is too small.");
	bufferHeader->m_copy_loaded_flag = 0;
	TrackedObjectsArray& trackedObjectsMap = bufferHeader->m_trackedObjects;

	// cleanup objects
	for( TrackedObjectsArray::iterator it = trackedObjectsMap.begin();
		it < trackedObjectsMap.end(); it++ )
	{
		const hkTypeInfo* typeInfo = it->m_typeInfo;
		void* objAddress = hkAddByteOffset(buffer, it->m_offset);
		typeInfo->cleanupLoadedObject(objAddress);
	}
	bufferHeader->~InplaceDataHeader();
}

void hkNativePackfileUtils::unloadInPlace( void* buffer, int bufferSize )
{
	hkPackfileHeader* packfileHeader = reinterpret_cast<hkPackfileHeader*>( buffer );
	HK_ASSERT2(0x6c1adf70, (packfileHeader->m_magic[0] == 0x57e0e057) && (packfileHeader->m_magic[1] == 0x10c0c010), "Invalid packfile buffer");
	HK_ASSERT2(0x673e19d4, hkSizeOf(hkPackfileHeader) <= bufferSize, "The buffer is too small.");
	HK_ASSERT2(0x64b400a8, packfileHeader->m_flags & 1,"Packfile was not loaded in-place");
	
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		hkPackfileSectionHeader& inSection = *packfileHeader->getSectionHeader(buffer, sectionIndex);
		if(inSection.getFinishSize())
		{
			const int* virtualFixups = reinterpret_cast<const int*>( hkAddByteOffsetConst( buffer, inSection.m_absoluteDataStart + inSection.m_virtualFixupsOffset ) );

			for( int i = 0; i < inSection.getFinishSize() / hkSizeOf(hkInt32); i += 3 )
			{
				int srcOff = virtualFixups[i  ];
				if( srcOff == -1 ) continue;
				HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );

				// automatically checks for dest section loaded
				void* objAddress = hkAddByteOffset(hkAddByteOffset(buffer, inSection.m_absoluteDataStart ), srcOff);
				hkUlong tempPointer = static_cast<hkUlong>(((static_cast<hkUint64>(virtualFixups[i+1]) << 32) & 0xFFFFFFFF00000000LL) |
															(static_cast<hkUint64>(virtualFixups[i+2]) & 0x00000000FFFFFFFFLL));

				const hkTypeInfo* registeredType = reinterpret_cast<const hkTypeInfo*>(tempPointer);
				if (registeredType)
				{
					registeredType->cleanupLoadedObject(objAddress);
				}
			}
		}
	}
}

void hkNativePackfileUtils::getImportsExports(const void* loadedBuffer, hkArray<hkResource::Export>& expOut, hkArray<hkResource::Import>& impOut )
{
	const InplaceDataHeader* bufferHeader = reinterpret_cast<const InplaceDataHeader*>( loadedBuffer );
	HK_ASSERT2(0x53a163e1, bufferHeader->m_copy_loaded_flag == COPY_LOADED_FLAG, "Invalid packfile buffer");

	for( int sectionIndex = 0; sectionIndex < bufferHeader->m_sections.getSize(); ++sectionIndex )
	{
		const hkPackfileSectionHeader& section = bufferHeader->m_sections[sectionIndex];
		const void* sectionBegin = hkAddByteOffsetConst(loadedBuffer, section.m_absoluteDataStart);
		section.getExports(sectionBegin, expOut);
		section.getImports(sectionBegin, impOut);
	}
}

void HK_CALL getImportsExportsInPlace(const void* loadedBuffer, hkArray<hkResource::Export>& expOut, hkArray<hkResource::Import>& impOut )
{
	const hkPackfileHeader* packfileHeader = reinterpret_cast<const hkPackfileHeader*>( loadedBuffer );
	HK_ASSERT2(0x47c8d988, (packfileHeader->m_magic[0] == 0x57e0e057) && (packfileHeader->m_magic[1] == 0x10c0c010), "Invalid packfile buffer");
	HK_ASSERT2(0x47c8d988, (packfileHeader->m_flags & 1), "Packfile buffer not loaded in place");
	
	hkPackfileSectionHeader* inSections = packfileHeader->m_numSections > 0 ? (hkPackfileSectionHeader*)(packfileHeader + 1) : HK_NULL;
	
	for( int sectionIndex = 0; sectionIndex < packfileHeader->m_numSections; ++sectionIndex )
	{
		const hkPackfileSectionHeader& section = inSections[sectionIndex];
		const void* sectionBegin = hkAddByteOffsetConst(loadedBuffer, section.m_absoluteDataStart);
		section.getExports(sectionBegin, expOut);
		section.getImports(sectionBegin, impOut);
	}	
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
