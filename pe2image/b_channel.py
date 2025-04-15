import os
from . import utils
import pefile

def apply_section_permissions_to_b(pe_image) -> None:
    """score B channel using section metaData
    
    """
    # definition of known section/packer's section
    whitelist_names = [b'.text', b'.rdata', b'.data', b'.rsrc', b'.reloc',
                       b'.idata', b'.edata', b'.pdata', b'.bss', b'.CRT', b'.tls']
    packer_names = [b'UPX', b'ASPack', b'PECompact', b'MPRESS', b'kkrunchy']
    
    num_sections = len(pe_image.pe.sections)
    if hasattr(pe_image.pe, 'DIRECTORY_ENTRY_IMPORT'):
        import_count = sum(len(entry.imports) for entry in pe_image.pe.DIRECTORY_ENTRY_IMPORT)
    else:
        import_count = 0

    # calculate overlay section
    if pe_image.pe.sections:
        end_of_last_section = max(s.PointerToRawData + s.SizeOfRawData for s in pe_image.pe.sections)
    else:
        end_of_last_section = 0
    file_size = len(pe_image.pe.__data__)
    overlay_size = file_size - end_of_last_section if file_size > end_of_last_section else 0

    # decision for section's index including entry point
    entry_point_rva = pe_image.pe.OPTIONAL_HEADER.AddressOfEntryPoint
    ep_section_index = None
    for idx, section in enumerate(pe_image.pe.sections):
        start = section.VirtualAddress
        end = start + max(section.Misc_VirtualSize, section.SizeOfRawData)
        if start <= entry_point_rva < end:
            ep_section_index = idx
            break

    log_lines = []
    log_lines.append(f"Total Sections: {num_sections}, Import Count: {import_count}, Overlay Size: {overlay_size}\n")

    # score each section's score
    for idx, section in enumerate(pe_image.pe.sections):
        name_bytes = section.Name.strip(b'\x00')
        try:
            name_str = name_bytes.decode('utf-8')
        except UnicodeDecodeError:
            name_str = str(name_bytes)
        flags = section.Characteristics
        has_x = bool(flags & 0x20000000)  # Execute
        has_w = bool(flags & 0x80000000)  # Write
        has_r = bool(flags & 0x40000000)  # Read
        
        score = 0
        details = []  # log score
        
        # (a) score based on authority
        if has_x and has_w:
            score += 100
            details.append("Exec+Write:+100")
        elif has_x and not has_w:
            if name_bytes in whitelist_names:
                score += 5
                details.append("Exec only (.text):+5")
            else:
                score += 20
                details.append("Exec only (non-whitelist):+20")
        elif has_w and has_r and not has_x:
            score += 30
            details.append("Write+Read:+30")
        elif has_r and not has_x and not has_w:
            score += 10
            details.append("Read only:+10")
        
        # (b) score based on section name
        if any(pat in name_bytes for pat in packer_names):
            score += 80
            details.append("Packer name detected:+80")
        elif name_bytes not in whitelist_names:
            if not name_str or any(ord(ch) < 0x20 for ch in name_str):
                score += 60
                details.append("Invalid name:+60")
            else:
                score += 50
                details.append("Non-whitelisted name:+50")
        else:
            details.append("Whitelisted name:+0")
        
        # (c) score based on entropy
        data = section.get_data()
        ent = utils.compute_entropy(data)
        if ent > 7.5:
            score += 70
            details.append(f"Entropy({ent:.2f})>7.5:+70")
        elif ent > 7.0:
            score += 40
            details.append(f"Entropy({ent:.2f})>7.0:+40")
        elif ent > 6.0:
            score += 10
            details.append(f"Entropy({ent:.2f})>6.0:+10")
        else:
            details.append(f"Entropy({ent:.2f}):+0")
        
        # (d) score based on section size
        raw_size = section.SizeOfRawData
        virt_size = section.Misc_VirtualSize
        if raw_size == 0 and virt_size > 0:
            score += 30
            details.append("Raw=0, Virt>0:+30")
        if raw_size > 0 and virt_size > raw_size * 1.2:
            score += 30
            details.append("Virt >1.2*Raw:+30")
        
        # (e) score based on entrypoint
        if idx == ep_section_index:
            if not has_x:
                score += 50
                details.append("Entry section, no Exec:+50")
            if name_bytes not in whitelist_names:
                score += 40
                details.append("Entry section, non-whitelist:+40")
            if ent > 7.0:
                score += 40
                details.append("Entry section, high entropy:+40")
        
        # (f) score based on number of sections
        if num_sections <= 2:
            score += 40
            details.append("Sections<=2:+40")
        elif num_sections == 3:
            score += 20
            details.append("Sections==3:+20")
        
        # (g) score based on importing count
        if import_count == 0:
            if has_x or (idx == ep_section_index):
                score += 80
                details.append("No imports, Exec/Entry:+80")
        elif import_count < 5:
            if has_x or (idx == ep_section_index):
                score += 50
                details.append("Imports<5, Exec/Entry:+50")
        elif import_count < 10:
            if has_x or (idx == ep_section_index):
                score += 20
                details.append("Imports<10, Exec/Entry:+20")
        
        # (h) other factor
        packer_signals = 0
        # 1) abnormal section name with execute authority
        if name_bytes not in whitelist_names and has_x:
            packer_signals += 1
        # 2) strange Raw size / Virtual size
        if raw_size == 0 and virt_size > 0:
            packer_signals += 1
        elif raw_size > 0 and (virt_size / raw_size) > 5:
            packer_signals += 1
        # 3) high entropy
        if ent > 7.0:
            packer_signals += 1
        # 4) section name pattern related to packers
        if any(pat in name_bytes for pat in packer_names):
            packer_signals += 2
        
        if packer_signals >= 3:
            score += 40
            details.append("Packer suspicion:+40")
        
        # 0 <= score <= 255
        score = max(0, min(255, score))
        
        # B channel mapping
        start_offset = section.PointerToRawData
        end_offset = start_offset + section.SizeOfRawData
        if start_offset < pe_image.pe_rgb_map.shape[0]:
            end_offset = min(end_offset, pe_image.pe_rgb_map.shape[0])
            pe_image.pe_rgb_map[start_offset:end_offset, 2] = score
        
        # log
        log_lines.append(
            f"Section {idx} ({name_str}): score={score}, start={start_offset}, end={end_offset}, "
            f"flags=0x{flags:08X}, entropy={ent:.2f}, raw_size={raw_size}, virt_size={virt_size}\n"
            f"  Details: {', '.join(details)}\n"
        )
    
    # (i) overlay
    if overlay_size > 0:
        overlay_data = pe_image.pe.__data__[end_of_last_section:]
        overlay_entropy = utils.compute_entropy(overlay_data)
        # check certification directory
        cert_dir = pe_image.pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']]
        has_cert = cert_dir.Size > 0 if cert_dir else False

        # check signature
        known_signature = any(sig in overlay_data for sig in packer_names)
        
        # check entropy only if there's no certification
        if (overlay_entropy > 7.0 or known_signature) and not has_cert:
            overlay_score = 200
            pe_image.pe_rgb_map[end_of_last_section:pe_image.pe_rgb_map.shape[0], 2] = overlay_score
            conditions = []
            if overlay_entropy > 7.0:
                conditions.append(f"high entropy ({overlay_entropy:.2f})")
            if known_signature:
                conditions.append("known packer signature detected")
            condition_details = " and ".join(conditions)
            log_lines.append(f"Overlay detected (size={overlay_size}, entropy={overlay_entropy:.2f}): {condition_details} -> assigned score {overlay_score}\n")
            if pe_image.verbose:
                print(f"[INFO] Suspicious overlay of {overlay_size} byte(s) with {condition_details} detected, assigned score {overlay_score}.")
        else:
            log_lines.append(f"Overlay detected (size={overlay_size}, entropy={overlay_entropy:.2f}): no suspicious indicators, score 0\n")
    
    # save log
    log_dir = "output/log"
    os.makedirs(log_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pe_image.file_path))[0]
    log_path = os.path.join(log_dir, f"{base_name}_B_channel.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)
