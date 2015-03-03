/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileWriter.h>
#include <Common/Serialize/Serialize/Xml/hkXmlObjectWriter.h>

#if 0
extern "C" int printf(const char*,...);
#	define PRINT(A) printf A
#else
#	define PRINT(A) /* nothing */
#endif

// Version 6:
//   Objects are saved in sort-order based on object pointer dependencies.
//   hkClass for homogeneous array is stored in the same section as an object containing the array
//   (the hkClass is not embedded, but referenced).
//   <packfile> tag has 'toplevelobject' attribute.
// Version 7:
//   Synced with binary version: 2-byte alignment for c-strings.
// Version 8:
//	 Update hkClass signature
// Version 9:
//   Supports tuples of tuples
// Version 10:
//   Adding the array of satisfied predicates in the header.

	// Change the detection code in hkSerializeUtil.detectFormat.switch if you bump this number
	// Also we generally it in sync with hkBinaryPackfileWriter::CURRENT_FILE_VERSION
const hkUint32 hkXmlPackfileWriter::CURRENT_FILE_VERSION = 11;

hkXmlPackfileWriter::hkXmlPackfileWriter(const Options & options)
: hkPackfileWriter(options)
{
	// Types must come first since we cannot read objects
	// without their type info and we cannot seek to find it.
	addSection( SECTION_TAG_TYPES );
	setSectionForClass(hkClassClass, SECTION_TAG_TYPES);
	setSectionForClass(hkClassMemberClass, SECTION_TAG_TYPES);
	setSectionForClass(hkClassEnumClass, SECTION_TAG_TYPES);
	setSectionForClass(hkClassEnumItemClass, SECTION_TAG_TYPES);
}

namespace
{
	struct PackfileNameFromAddress : public hkXmlObjectWriter::NameFromAddress
	{
		typedef hkPointerMap<const void*, int> IntFromPtrMap;
		typedef hkPointerMap<const void*, const void*> PtrFromPtrMap;
		typedef hkPointerMap<const void*, const char*> StrFromPtrMap;

		const IntFromPtrMap& m_knownObjects;
		const PtrFromPtrMap& m_replacements;
		const StrFromPtrMap& m_imports;
		IntFromPtrMap  m_indexFromAddress;

		PackfileNameFromAddress(
			const IntFromPtrMap& known,
			const PtrFromPtrMap& replace,
			const StrFromPtrMap& imports ) :
			m_knownObjects(known),
			m_replacements(replace),
			m_imports(imports)
		{
		}

		void ignore( const void* p )
		{
			m_indexFromAddress.insert(p,0);
		}

		int nameFromAddress(const void* address, char* name, int nameLen )
		{
			int index = 0;
			if( address )
			{
				while( const void* replacement = m_replacements.getWithDefault(address, HK_NULL) )
				{
					address = replacement;
				}
				const char* importName = m_imports.getWithDefault(address, HK_NULL);
				if( importName )
				{
					hkString::strNcpy( name, "@", nameLen );
					hkString::strNcpy( name+1, importName, nameLen-1 );
					return hkString::strLen(name);
				}

				index = m_indexFromAddress.getWithDefault( address, -1 );
				if( index == -1 )
				{
					// have not seen this pointer, should we ignore it?
					if( m_knownObjects.getWithDefault(address, 0) == -1 /*hkPackfileWriter::INDEX_IGNORE*/ )
					{
						index = 0;
					}
					else // create a new id
					{
						index = m_indexFromAddress.getSize() + 1;
						m_indexFromAddress.insert( address, index );
					}
				}
			}
			if( index )
			{
				return hkString::snprintf(name, nameLen, "#%04i", index);
			}
			else
			{
				hkString::strNcpy(name, "null", nameLen);
				return 4;
			}
		}
	};
}

static hkBool32 savePendingData( hkXmlObjectWriter& writer, hkStreamWriter* stream, hkOstream& os, hkArray<hkXmlPackfileWriter::PendingWrite>& pendingWrites, int pendingIndex, int sectionIndex, hkSerializeMultiMap<const void*, int>& pwIndexesFromReferencedPointer, hkPointerMap<const void*, const char*>& exports, PackfileNameFromAddress& namer, hkPointerMap<hkUlong, hkBool32>& donePWIndexes );

hkResult hkXmlPackfileWriter::save( hkStreamWriter* stream, const Options& options )
{
	PackfileNameFromAddress namer( m_knownObjects, m_replacements, m_imports );
	hkXmlObjectWriter writer(namer, options.m_writeSerializedFalse);
	int classSectionIndex = static_cast<int>( sectionTagToIndex( SECTION_TAG_TYPES ) );
	
	// maybe skip all class objects
	if( options.m_writeMetaInfo == false )
	{
		for( int pendingIndex = 0; pendingIndex < m_pendingWrites.getSize(); ++pendingIndex )
		{
			PendingWrite& pending = m_pendingWrites[pendingIndex];
			if( pending.m_sectionIndex == classSectionIndex )
			{
				HK_ASSERT(0x56f54de3, m_contentsPtrPWIndex != pendingIndex);
				namer.ignore( pending.m_origPointer );
				if( pending.m_origPointer != pending.m_pointer )
				{
					namer.ignore( pending.m_pointer );
				}
			}
		}
	}

	// packfile header
	hkOstream os(stream);
	{
		os.printf("<?xml version=\"1.0\" encoding=\"ascii\"?>\n");
		const char* contentsversion = options.m_contentsVersion;
		char defaultContentsVersion[32];
		if( contentsversion == HK_NULL )
		{
			getCurrentVersion(defaultContentsVersion,sizeof(defaultContentsVersion));
			contentsversion = defaultContentsVersion;
		}
		char namebuffer[256];
		HK_ASSERT(0x56f54de2, m_contentsPtrPWIndex >=0 && m_contentsPtrPWIndex < m_pendingWrites.getSize() && m_pendingWrites[m_contentsPtrPWIndex].m_origPointer != HK_NULL );
		namer.nameFromAddress( m_pendingWrites[m_contentsPtrPWIndex].m_origPointer, namebuffer, sizeof(namebuffer) );
		os.printf("<hkpackfile classversion=\"%d\" contentsversion=\"%s\" toplevelobject=\"%s\" maxpredicate=\"%d\" predicates=\"",
			CURRENT_FILE_VERSION, contentsversion, namebuffer, 0); 
		os.printf("\">\n");
		writer.adjustIndent(1);
	}

	// write comments about ignored objects
	for( int skippedIndex = 0; skippedIndex < m_objectsWithUnregisteredClass.getSize(); ++skippedIndex )
	{
		hkVariant& v = m_objectsWithUnregisteredClass[skippedIndex];
		os.printf("\t<!-- Skipped %s at address %p -->\n", v.m_class->getName(), v.m_object );
	}
	
	// body of the packfile
	hkPointerMap<hkUlong, hkBool32> donePWIndexes;
	for( int sectionIndex = 0; sectionIndex < m_knownSections.getSize(); ++sectionIndex )
	{
		if( options.m_writeMetaInfo || sectionIndex != classSectionIndex )
		{
			os.printf("\n\t<hksection name=\"%s\">\n", m_knownSections[sectionIndex]);
			writer.adjustIndent(1);
			for( int pendingIndex = m_pendingWrites.getSize()-1; pendingIndex >= 0; --pendingIndex )
			{
				savePendingData(writer, stream, os, m_pendingWrites, pendingIndex, sectionIndex, m_pwIndexesFromReferencedPointer, m_exports, namer, donePWIndexes);
			}
			writer.adjustIndent(-1);
			os.printf("\n\t</hksection>");
		}
	}

	// footer
	writer.adjustIndent(-1);
	os.printf("\n\n</hkpackfile>\n");
	stream->flush();
	
	return stream->isOk() ? HK_SUCCESS : HK_FAILURE;
}

static hkBool32 savePendingData( hkXmlObjectWriter& writer, hkStreamWriter* stream, hkOstream& os, hkArray<hkXmlPackfileWriter::PendingWrite>& pendingWrites, int pendingIndex, int sectionIndex, hkSerializeMultiMap<const void*, int>& pwIndexesFromReferencedPointer, hkPointerMap<const void*, const char*>& exports, PackfileNameFromAddress& namer, hkPointerMap<hkUlong, hkBool32>& donePWIndexes )
{
	if( donePWIndexes.hasKey(pendingIndex) )
	{
		return true;
	}
	hkXmlPackfileWriter::PendingWrite& pending = pendingWrites[pendingIndex];
	char namebuffer[256];
	namer.nameFromAddress( pending.m_origPointer, namebuffer, sizeof(namebuffer) );
	if( pending.m_sectionIndex == sectionIndex )
	{
		donePWIndexes.insert(pendingIndex, true);
		for( int keyindex = pwIndexesFromReferencedPointer.getFirstIndex(pending.m_origPointer); keyindex != -1; )
		{
			int pwi = pwIndexesFromReferencedPointer.getValue(keyindex);

			char pwiname[256];
			namer.nameFromAddress( pendingWrites[pwi].m_origPointer, pwiname, sizeof(pwiname) );

			PRINT(("->%d\t%s (0x%p) [%d] (%d) -> %s (0x%p) [%d] (%d)%s\n", sectionIndex, namebuffer, pending.m_origPointer, pendingIndex, pending.m_sectionIndex, pwiname, pendingWrites[pwi].m_origPointer, pwi, pendingWrites[pwi].m_sectionIndex, donePWIndexes.hasKey(pwi) ? "" : "..."));
			if( savePendingData(writer, stream, os, pendingWrites, pwi, sectionIndex, pwIndexesFromReferencedPointer, exports, namer, donePWIndexes) )
			{
				keyindex = pwIndexesFromReferencedPointer.removeByIndex(pending.m_origPointer, keyindex);
			}
			else
			{
				keyindex = pwIndexesFromReferencedPointer.getNextIndex(keyindex);
			}
		}

		PRINT(("\t%s (0x%p) [%d] (%d)\n", namebuffer, pending.m_origPointer, pendingIndex, pending.m_sectionIndex));
		const char* attributes[5];

		char sigbuffer[2+8+1]; // (0x) + (ffffffff) + (nil)
		hkString::snprintf(sigbuffer, sizeof(sigbuffer), "%#08x", pending.m_klass->getSignature() );
		int attridx = 0;
		attributes[attridx++] = "signature";
		attributes[attridx++] = sigbuffer;
		if( const char* exported = exports.getWithDefault(pending.m_pointer, HK_NULL) )
		{
			attributes[attridx++] = "export";
			attributes[attridx++] = exported;
		}
		attributes[attridx++] = 0;
		writer.writeObjectWithElement( stream, pending.m_pointer, *pending.m_klass, namebuffer, attributes );
		os.printf("\n");
		return true;
	}
	else
	{
		if( hkString::strCmp(namebuffer, "null") == 0 )
		{
			PRINT(("\t%s (%d)\n", namebuffer, pending.m_sectionIndex));
			donePWIndexes.insert(pendingIndex, false);
			return true;
		}
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
