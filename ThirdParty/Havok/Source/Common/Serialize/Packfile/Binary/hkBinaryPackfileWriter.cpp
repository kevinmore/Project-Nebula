/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/Writer/SubStream/hkSubStreamWriter.h>
#include <Common/Serialize/Packfile/Binary/hkBinaryPackfileWriter.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileHeader.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>
#include <Common/Serialize/Serialize/Platform/hkPlatformObjectWriter.h>

HK_COMPILE_TIME_ASSERT( sizeof(hkPackfileHeader) == 16*sizeof(hkInt32));
HK_COMPILE_TIME_ASSERT( sizeof(hkPackfileSectionHeader) == 16*sizeof(hkInt32));

// Version 1:
//   hkClass with m_hasVtable member.
//   m_header.m_contents*{Index,Offset} == -1
// Version 2:
//   hkClass with m_defaults member instead of m_hasVtable.
//   m_header.m_contents*{Index,Offset} set to index/offset of CLASS of contents
//   __classindex__ and __dataindex__ sections for indexing contents
// Version 3:
//   m_header.m_contents*{Index,Offset} set to index/offset of CLASS NAME of contents
//   4.x beta1 added finish fixups for all objects, not just virtual
// Version 4:
//   finish fixups for all objects, not just virtual
//   packfilesectionheader.external renamed to exports
//   packfilesectionheader.imports added in place of endOffset
//   packfilesectionheader.endOffset moved to spare
//   (opt) no longer store __dataindex__ and __classindex__ - reconstruct from finish fixups
//   (opt) store the class signature (4 byte crc32 + magic '\t') before the classname in __classnames__ section.
// Version 5:
//   hkClass and hkClassMember get new attributes members for custom metadata.
//   hkClass has new hkFlags member.
//   hkEnum and hkFlags sizes are in m_subType, not in flags
//   hkClass TYPE_ZERO replaced by SERIALIZE_IGNORED flag
// Version 6:
//   hkClass for homogeneous array is stored in the same section as an object containing the array, but not in
//   __types__ section.
//   hkClass version is now used.
// Version 7:
//   All c-strings are stored as 2-byte aligned.
// Version 8:
//	 hkClassMember updated for new hkRelArray type
// Version 9:
//   hkTypeManager type support now allows tuples of tuples. 
// Version 10:
//   Adding the array of satisfied predicates in the header.
// Version 11:
//   Modified the hkPackfileSectionHeader layout adding some padding to make sure HK_REAL_ALIGNMENT is respected
//   even in double precision builds. Prior to this version, saving/loading of double precious packfiles wasn't working 
//   properly.

	// Change the detection code in hkSerializeUtil.detectFormat.switch if you bump this number
	// Also we generally keep it in sync with hkXmlPackfileWriter::CURRENT_FILE_VERSION.
const hkUint32 hkBinaryPackfileWriter::CURRENT_FILE_VERSION = 11;

// Class names used for finish object fixups.
hkPackfileWriter::SectionTag hkBinaryPackfileWriter::SECTION_TAG_CLASSNAMES = "__classnames__";

// match with reader
#define INDEX_CLASSNAMES 0
#define INDEX_FIRST_NORMAL 1

namespace
{
	struct WriteLocation
	{
		int m_sectionIndex;
		int m_offset;
	};

	template <typename T>
	int getArraySizeInBytes( const hkArray<T>& a )
	{
		return a.getSize() * sizeof(T);
	}
}

// data to pass around from save function
struct hkBinaryPackfileWriter::SaveContext
{
	SaveContext( const hkPackfileWriter::Options& options)
		:	m_options(options),
			m_writer(options.m_layout),
			m_endianSwapRequired( hkStructureLayout::HostLayoutRules.m_littleEndian != options.m_layout.getRules().m_littleEndian )
	{
	}

	const hkPackfileWriter::Options& m_options;
	hkArray<hkPackfileSectionHeader> m_sectionHeaders;
	hkArray<WriteLocation> m_locations; // eventual locations of objects - matches m_pendingwrites
	hkStreamWriter* m_stream;
	hkPlatformObjectWriter m_writer;
	hkStringMap<int> m_usedClassNames;
	hkArray<hkRelocationInfo> m_relocsBySect;
	hkBool m_endianSwapRequired;
};

HK_COMPILE_TIME_ASSERT( sizeof(hkStructureLayout::LayoutRules)==4 );
HK_COMPILE_TIME_ASSERT( sizeof(hkPackfileHeader)==64 );
HK_COMPILE_TIME_ASSERT( sizeof(hkPackfileSectionHeader)==64 );

static int padUp( hkStreamWriter* w, int pad=HK_REAL_ALIGNMENT, unsigned char padData = 0xff )
{
	int padding = 0;
	int o = w->tell();
	hkInplaceArray<char, 32> buf;
	buf.setSize(pad, padData);
	if( o & (pad-1) )
	{
		padding = pad - (o&(pad-1));
		w->write( buf.begin(), padding);
	}
	return padding;
}

hkBinaryPackfileWriter::hkBinaryPackfileWriter(const Options& options)
: hkPackfileWriter(options)
{
	// ensure we have a classname section first
	addSection(SECTION_TAG_CLASSNAMES);
	setSectionForClass(hkClassClass, SECTION_TAG_TYPES);
	setSectionForClass(hkClassMemberClass, SECTION_TAG_TYPES);
	setSectionForClass(hkClassEnumClass, SECTION_TAG_TYPES);
	setSectionForClass(hkClassEnumItemClass, SECTION_TAG_TYPES);
}

hkBinaryPackfileWriter::~hkBinaryPackfileWriter()
{
}

void hkBinaryPackfileWriter::fillSectionTags( SaveContext& context )
{
	hkArray<hkPackfileSectionHeader>& sectHead = context.m_sectionHeaders;
	sectHead.setSize( m_knownSections.getSize() );
	hkString::memSet( sectHead.begin(), -1, getArraySizeInBytes(sectHead) );

	for( int i = 0; i < m_knownSections.getSize(); ++i )
	{
		hkString::strNcpy( sectHead[i].m_sectionTag, m_knownSections[i], sizeof(sectHead[i].m_sectionTag));
	}
}

static void saveFileHeader(hkPlatformObjectWriter& writer, hkStreamWriter* stream, const hkPackfileWriter::Options& options, int numSections )
{
	hkPackfileHeader header;
	header.m_userTag = options.m_userTag;
	header.m_fileVersion = hkBinaryPackfileWriter::CURRENT_FILE_VERSION;
	hkString::memCpy( header.m_layoutRules, &options.m_layout, sizeof(hkStructureLayout::LayoutRules));
	header.m_numSections = numSections;
	hkString::memSet(header.m_pad, -1, sizeof(header.m_pad));
	if( options.m_contentsVersion == HK_NULL )
	{
		hkPackfileWriter::getCurrentVersion(header.m_contentsVersion, sizeof(header.m_contentsVersion));
	}
	else
	{
		hkString::strNcpy( header.m_contentsVersion, options.m_contentsVersion, sizeof(header.m_contentsVersion)); 
	}
	hkRelocationInfo fixups;
	writer.writeObject( stream, &header, hkPackfileHeaderClass, fixups );
}

void hkBinaryPackfileWriter::writeClassName( SaveContext& context, const hkClass& k, int absoluteDataStart )
{
	// signature (5 bytes)
	int sig = k.getSignature();
	if( context.m_endianSwapRequired )
	{
		union { hkInt32 i; char c[4]; } u;
		u.i = sig;
		char t0 = u.c[0]; u.c[0] = u.c[3]; u.c[3] = t0;
		char t1 = u.c[1]; u.c[1] = u.c[2]; u.c[2] = t1;
		sig = u.i;
	}
	struct { hkInt32 sig; char marker; } s;
	s.marker = '\t';
	s.sig = sig;
	context.m_stream->write( &s, sizeof(hkInt32)+sizeof(char) );

	// class name
	const char* name = k.getName();
	context.m_usedClassNames.insert( name, context.m_stream->tell() - absoluteDataStart );
	context.m_stream->write( name, hkString::strLen(name) + 1 );
}

void hkBinaryPackfileWriter::writeAllObjects( SaveContext& context )
{
	// first section is the classname section
	if(1)
	{
		hkPackfileSectionHeader& section = context.m_sectionHeaders[INDEX_CLASSNAMES];
		section.m_absoluteDataStart = context.m_stream->tell();

		hkStringMap<hkBool32> writtenClassnames;
		// special case for hkClass and hkClassEnum names, by default these are ignored
		const hkClass* specialClasses[] = {&hkClassClass, &hkClassMemberClass, &hkClassEnumClass, &hkClassEnumItemClass};
		for( int i = 0; i < int(sizeof(specialClasses)/sizeof(const char*)); i++ )
		{
			writtenClassnames.insert(specialClasses[i]->getName(), true);
			writeClassName( context, *specialClasses[i], section.m_absoluteDataStart );
		}

		for( int i = 0; i < m_pendingWrites.getSize(); ++i )
		{
			HK_ASSERT(0x0897fde3, m_pendingWrites[i].m_klass);
			if( writtenClassnames.hasKey(m_pendingWrites[i].m_klass->getName()) )
			{
				continue;
			}
			writtenClassnames.insert(m_pendingWrites[i].m_klass->getName(), true);
			writeClassName( context, *m_pendingWrites[i].m_klass, section.m_absoluteDataStart );
		}

		padUp(context.m_stream);
		int fpos = context.m_stream->tell() - section.m_absoluteDataStart;
		section.m_localFixupsOffset = fpos;
		section.m_globalFixupsOffset = fpos;
		section.m_virtualFixupsOffset = fpos;
		section.m_exportsOffset = fpos;
		section.m_importsOffset = fpos;
		section.m_endOffset = fpos;
	}

	context.m_locations.setSize( m_pendingWrites.getSize() );
	context.m_relocsBySect.setSize( context.m_sectionHeaders.getSize() );

	// pass one - traverse and save all objects
	int classSectionIndex = sectionTagToIndex(SECTION_TAG_TYPES);
	for( int sectionIndex = INDEX_FIRST_NORMAL; sectionIndex < context.m_sectionHeaders.getSize(); ++sectionIndex)
	{
		hkPackfileSectionHeader& section = context.m_sectionHeaders[sectionIndex];
		section.m_absoluteDataStart = context.m_stream->tell();
		hkRelocationInfo fixups;

		// Go through all the pending writes, and save all the writes for this particular section
		if( context.m_options.m_writeMetaInfo || sectionIndex != classSectionIndex )
		{
			for( int i = 0; i < m_pendingWrites.getSize(); ++i )
			{
				const hkBinaryPackfileWriter::PendingWrite& w = m_pendingWrites[i];
				HK_ASSERT(0x309db72b,w.m_sectionIndex != 0);
				if( w.m_sectionIndex == sectionIndex )
				{
					const void* dataStart = w.m_pointer;
					const hkClass* klass = w.m_klass;

					context.m_locations[i].m_sectionIndex = sectionIndex;
					context.m_locations[i].m_offset = context.m_stream->tell() - section.m_absoluteDataStart;

					if( klass )
					{
						context.m_writer.writeObject(context.m_stream, dataStart, *klass, fixups );
					}
					else
					{
						HK_ASSERT2(0x44d609be,0, "No class in requested write to a packfile section. Raw write will have to be performed." );
						context.m_stream->write( w.m_pointer, w.m_dataSize );
					}
				}
			}
		}
		else // class section
		{
			for( int i = 0; i < m_pendingWrites.getSize(); ++i )
			{
				const hkBinaryPackfileWriter::PendingWrite& w = m_pendingWrites[i];
				HK_ASSERT(0x7216e4d5,w.m_sectionIndex != 0);
				if( w.m_sectionIndex == classSectionIndex )
				{
					context.m_locations[i].m_sectionIndex = -1;
					context.m_locations[i].m_offset = -1;
				}
			}
		}

		// local fixups are relative to entire file - make them section relative.
		{
			for( int i = 0; i < fixups.m_local.getSize(); ++i )
			{
				fixups.m_local[i].m_fromOffset -= section.m_absoluteDataStart;
				fixups.m_local[i].m_toOffset -= section.m_absoluteDataStart;
			}
		}
		// some global fixups are changed into imports
		{
			for( int i = fixups.m_global.getSize() - 1; i >= 0; --i )
			{
				if( const char* id = m_imports.getWithDefault( fixups.m_global[i].m_toAddress, HK_NULL ) )
				{
					const hkRelocationInfo::Global& fix = fixups.m_global[i];
					HK_ASSERT3(0x68eb4f33, m_knownObjects.getWithDefault(fixups.m_global[i].m_toAddress, -1) < 0,
						"Object marked as imported, but is in file " << fix.m_toAddress << " " << fix.m_toClass->getName() );
					hkRelocationInfo::Import& imp = fixups.m_imports.expandOne();
				   	imp.m_fromOffset = fix.m_fromOffset - section.m_absoluteDataStart;
					imp.m_identifier = id;
					fixups.m_global.removeAt(i);
				}
			}
		}

		// some global fixups actually refer to this section. make them local.
		// This defeats versioning code, so only switch on for baking.
		if(0)
		{
			int srci = 0;
			int dsti = 0;

			for( ; srci < fixups.m_global.getSize(); ++srci )
			{
				void* dstPtr = fixups.m_global[srci].m_toAddress;
				
				int idx = 0;
				if( m_knownObjects.get( dstPtr, &idx ) == HK_SUCCESS )
				{
					if( idx != INDEX_IGNORE )
					{
						if( sectionIndex == m_pendingWrites[idx].m_sectionIndex )
						{
							hkUlong srcOff = fixups.m_global[srci].m_fromOffset - section.m_absoluteDataStart;
							int dstOff = context.m_locations[idx].m_offset;
							//hkcout.printf("* %x -> %x\n", section.m_id, m_index[idx].m_section );				
							//hkcout.printf("  %x -> %x\n", srcOff, dstOff );
							fixups.addLocal( int(srcOff), int(dstOff) );
						}
						else // reference to other section
						{
							fixups.m_global[dsti] = fixups.m_global[srci];
							++dsti;
						}
					}
				}
				else
				{
					HK_ASSERT2(0x44d609be, 0, "Found a rogue pointer while writing a packfile section.");
				}
			}
			fixups.m_global.setSize( dsti );
		}

		hkOArchive oa(context.m_stream, context.m_endianSwapRequired );

		// local offsets
		section.m_localFixupsOffset = context.m_stream->tell() - section.m_absoluteDataStart;
		HK_COMPILE_TIME_ASSERT( sizeof(hkRelocationInfo::Local) == sizeof(int)*2 );
		oa.writeArrayGeneric( fixups.m_local.begin(), sizeof(int), 2*fixups.m_local.getSize() );
		padUp(context.m_stream);

		// global offsets - fix these up later when all offsets are known
		section.m_globalFixupsOffset = context.m_stream->tell() - section.m_absoluteDataStart;
		context.m_stream->seek( fixups.m_global.getSize()*3*sizeof(hkInt32), hkStreamWriter::STREAM_CUR ); // fixup later
		context.m_relocsBySect[sectionIndex].m_global.swap( fixups.m_global );
		padUp(context.m_stream);

		// vtable offsets  - fix these up later when all offsets are known
		section.m_virtualFixupsOffset = context.m_stream->tell() - section.m_absoluteDataStart;
		context.m_stream->seek( fixups.m_finish.getSize()*3*sizeof(hkInt32), hkStreamWriter::STREAM_CUR ); // fixup later
		context.m_relocsBySect[sectionIndex].m_finish.swap( fixups.m_finish );
		padUp(context.m_stream);

		// exports
		{
			section.m_exportsOffset = context.m_stream->tell() - section.m_absoluteDataStart;
			hkPointerMap<const void*, const char*>::Iterator it = m_exports.getIterator();
			bool someExports = false;
			for( ; m_exports.isValid(it); it = m_exports.getNext(it) )
			{
				const void* data = m_exports.getKey(it);
				int i = m_knownObjects.getWithDefault( data, -1 );
				if( i >= 0 && m_pendingWrites[i].m_sectionIndex == sectionIndex )
				{
					someExports = true;
					const char* name = m_exports.getValue(it);
					int off = context.m_locations[i].m_offset;
					oa.write32( off );
					oa.writeRaw( name, hkString::strLen(name)+1 );
					padUp(context.m_stream, 4);
				}
			}
			if( someExports )
			{
				oa.write32( -1 );
				padUp(context.m_stream);
			}
		}

		// imports
		{
			section.m_importsOffset = context.m_stream->tell() - section.m_absoluteDataStart;
			for( int i = 0; i < fixups.m_imports.getSize(); ++i )
			{
				const hkRelocationInfo::Import& imp = fixups.m_imports[i];
				oa.write32( imp.m_fromOffset );
				oa.writeRaw( imp.m_identifier, hkString::strLen(imp.m_identifier)+1 );
				padUp(context.m_stream, 4);
			}
			if( fixups.m_imports.getSize() )
			{
				oa.write32( -1 );
				context.m_relocsBySect[sectionIndex].m_imports.swap( fixups.m_imports );
				padUp(context.m_stream);
			}
		}

		//
		section.m_endOffset = context.m_stream->tell() - section.m_absoluteDataStart;
	}
}


void hkBinaryPackfileWriter::doDeferredWrites( SaveContext& context, int sectionHeadersStart )
{
	hkOArchive endianStream(context.m_stream, context.m_endianSwapRequired );

	// all pointers are now known so go back and fixup globals
	for( int sectionIndex = 0; sectionIndex < context.m_sectionHeaders.getSize(); ++sectionIndex)
	{
		// globals
		{
			hkPackfileSectionHeader& section = context.m_sectionHeaders[sectionIndex];
			const hkArray<hkRelocationInfo::Global>& globals = context.m_relocsBySect[sectionIndex].m_global;
			hkArray<hkUint32> fixups;
			fixups.setSize( 3 * globals.getSize(), hkUint32(-1) );
			for( int gi = 0, fi = 0 ; gi < globals.getSize(); gi += 1 )
			{
				int idx = m_knownObjects.getWithDefault( globals[gi].m_toAddress, INDEX_ERROR );
				//HK_ASSERT2( 0x585dc0db, idx != INDEX_ERROR, "Rogue pointer found during global fixups in packfile writer." );
				if( idx >= 0 && context.m_locations[idx].m_sectionIndex >= 0 )
				{
					fixups[fi  ] = globals[gi].m_fromOffset - section.m_absoluteDataStart;
					fixups[fi+1] = context.m_locations[idx].m_sectionIndex;
					fixups[fi+2] = context.m_locations[idx].m_offset;
					fi += 3;
				}
			}
			context.m_stream->seek( section.m_absoluteDataStart + section.m_globalFixupsOffset, hkStreamWriter::STREAM_SET );
			endianStream.writeArray32u( fixups.begin(), fixups.getSize() );
		}
		
		// finishers

		{
			hkPackfileSectionHeader& section = context.m_sectionHeaders[sectionIndex];
			const hkArray<hkRelocationInfo::Finish>& finish = context.m_relocsBySect[sectionIndex].m_finish;
			hkArray<hkUint32> fixups;
			fixups.setSize( 3* finish.getSize() );
			for( int gi = 0, fi = 0 ; gi < finish.getSize(); gi += 1, fi += 3 )
			{
				fixups[fi  ] = finish[gi].m_fromOffset - section.m_absoluteDataStart;
				fixups[fi+1] = 0;
				fixups[fi+2] = context.m_usedClassNames.getWithDefault( finish[gi].m_className, -1 );
				HK_ASSERT2( 0x585dc0dc, fixups[fi+2] != hkUint32(-1), "A fixup is corrupt during packfile write." );
			}
			context.m_stream->seek( section.m_absoluteDataStart + section.m_virtualFixupsOffset, hkStreamWriter::STREAM_SET );
			endianStream.writeArray32u( fixups.begin(), fixups.getSize() );
		}
	}

	// write sections
	{
		context.m_stream->seek( sectionHeadersStart, hkStreamWriter::STREAM_SET );
		for (int shi=0; shi < context.m_sectionHeaders.getSize(); ++shi)
		{
			hkRelocationInfo dummyReloc;
			context.m_writer.writeObject(context.m_stream, &context.m_sectionHeaders[shi], hkPackfileSectionHeaderClass, dummyReloc);
		}
	}

	// place the correct indices in the file header for the contents
	context.m_stream->seek( HK_OFFSET_OF(hkPackfileHeader, m_contentsSectionIndex), hkStreamWriter::STREAM_SET );
	const hkStructureLayout layout = context.m_writer.getLayout();
	hkBool endianSwap = (layout.getRules().m_littleEndian != layout.HostLayoutRules.m_littleEndian);
	hkOArchive outHeader( context.m_stream, endianSwap);
	outHeader.write32( context.m_locations[m_contentsPtrPWIndex].m_sectionIndex); 
	outHeader.write32( context.m_locations[m_contentsPtrPWIndex].m_offset);
	HK_ASSERT( 0x1961cd92, hkString::strCmp(m_pendingWrites[m_contentsClassPWIndex].m_klass->getName(),"hkClass") == 0 );
	const hkClass* topClass = static_cast<const hkClass*>(m_pendingWrites[m_contentsClassPWIndex].m_pointer);
	outHeader.write32( INDEX_CLASSNAMES );
	outHeader.write32( context.m_usedClassNames.getWithDefault(topClass->getName(),-1) ); 
}

// File structure.
//	hkPackFileHeader
//	hkPackFileSection * header.numSections
//  Section * header.numSections
hkResult hkBinaryPackfileWriter::save( hkStreamWriter* origStream, const Options& options )
{
	if( origStream->seekTellSupported() == false )
	{
		HK_WARN(0x3e6b8383, "hkBinaryPackfileWriter stream must support seek/tell.");
		return HK_FAILURE;
	}
	SaveContext context( options );
	hkSubStreamWriter subStream(origStream);
	hkStreamWriter* stream = &subStream;

	context.m_stream = stream;

	// set num sections and fill tags
	fillSectionTags( context );

	// save header
	saveFileHeader( context.m_writer, context.m_stream, options, context.m_sectionHeaders.getSize() );

	// section headers and index are just after the main header
	int sectionListStart = stream->tell();

	// leave space to write section descriptions
	stream->seek( getArraySizeInBytes( context.m_sectionHeaders), hkStreamWriter::STREAM_CUR );

	writeAllObjects( context );

	// remember end of file
	int endOfFile = stream->tell();

	// go back and write section fixups, section headers, index section
	doDeferredWrites( context, sectionListStart );

	// leave the file pointer where we expect it at the end of everything
	stream->seek( endOfFile, hkStreamWriter::STREAM_SET );
	stream->flush();

	return stream->isOk() ? HK_SUCCESS : HK_FAILURE;
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
